import functools
import os
import math
import importlib
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.cpp_extension import load
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from lightning.pytorch.strategies import DeepSpeedStrategy
from loguru import logger
from embed import VAE

if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

"""
Utility functions
"""


def __nop(ob):
    return ob


"""
Setup JIT
"""
BaseModule = nn.Module
JitFunction = __nop

if 'RWKV_JIT_ON' in os.environ and os.environ['RWKV_JIT_ON'] == '1':
    BaseModule = torch.jit.ScriptModule
    JitFunction = torch.jit.script_method

"""
CUDA compilation
"""
T_MAX = int(os.environ['RWKV_T_MAX'])
CUDA_SRC_PATH = f'{os.path.dirname(__file__)}/cuda'

"""
LoRa
"""
LORA_CONFIG = {
    "r": 8,
    "alpha": 32,
    "dropout": 0.001,
    "parts": {"att", "ffn", "ln", "time"},
}


class LoraLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool, lora_params: dict = LORA_CONFIG):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased LoraLinear not supported"

        r, alpha, dropout = lora_params["r"], lora_params["alpha"], lora_params["dropout"]
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)
        self.scaling = alpha / r

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (
            F.linear(x, self.weight) + self.scaling *
            F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B))


@functools.wraps(LoraLinear)
def make_linear_att(*args, **kwargs):
    if "att" in kwargs['lora_params']['parts'] and kwargs['lora_params']['r'] > 0:
        return LoraLinear(*args, **kwargs)
    else:
        return nn.Linear(*args, **kwargs)


@functools.wraps(LoraLinear)
def make_linear_ffn(*args, **kwargs):
    if "ffn" in kwargs['lora_params']['parts'] and kwargs['lora_params']['r'] > 0:
        return LoraLinear(*args, **kwargs)
    else:
        if 'lora_params' in kwargs:
            del kwargs['lora_params']

        return nn.Linear(*args, **kwargs)


if os.environ['RWKV_FLOAT_MODE'] == 'bf16':
    wkv_cuda = load(name=f'wkv_{T_MAX}_bf16', sources=[f'{CUDA_SRC_PATH}/wkv_op_bf16.cpp', f'{CUDA_SRC_PATH}/wkv_cuda_bf16.cu'], verbose=True, extra_cuda_cflags=[
                    '-t 4', '-std=c++17', '-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', '--extra-device-vectorization', f'-DTmax={T_MAX}'])

    class WKV(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v):
            ctx.B = B
            ctx.T = T
            ctx.C = C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w = -torch.exp(w.float().contiguous())
            u = u.bfloat16().contiguous()
            k = k.bfloat16().contiguous()
            v = v.bfloat16().contiguous()
            y = torch.empty((B, T, C), device=w.device,
                            memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            wkv_cuda.forward(B, T, C, w, u, k, v, y)
            ctx.save_for_backward(w, u, k, v, y)
            return y

        @staticmethod
        def backward(ctx, gy):
            B = ctx.B
            T = ctx.T
            C = ctx.C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w, u, k, v, y = ctx.saved_tensors
            gw = torch.empty((B, C), device=gy.device,
                             memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            gu = torch.empty((B, C), device=gy.device,
                             memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            gk = torch.empty((B, T, C), device=gy.device,
                             memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            gv = torch.empty((B, T, C), device=gy.device,
                             memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            wkv_cuda.backward(B, T, C, w, u, k, v, y,
                              gy.contiguous(), gw, gu, gk, gv)
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)
else:
    wkv_cuda = load(name=f'wkv_{T_MAX}', sources=[f'{CUDA_SRC_PATH}/wkv_op.cpp', f'{CUDA_SRC_PATH}/wkv_cuda.cu'], verbose=True, extra_cuda_cflags=[
                    '-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', '--extra-device-vectorization', f'-DTmax={T_MAX}'])

    class WKV(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v):
            ctx.B = B
            ctx.T = T
            ctx.C = C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            if '32' in os.environ['RWKV_FLOAT_MODE']:
                w = -torch.exp(w.contiguous())
                u = u.contiguous()
                k = k.contiguous()
                v = v.contiguous()
            else:
                w = -torch.exp(w.float().contiguous())
                u = u.float().contiguous()
                k = k.float().contiguous()
                v = v.float().contiguous()
            y = torch.empty((B, T, C), device=w.device,
                            memory_format=torch.contiguous_format)
            wkv_cuda.forward(B, T, C, w, u, k, v, y)
            ctx.save_for_backward(w, u, k, v, y)
            if '32' in os.environ['RWKV_FLOAT_MODE']:
                return y
            elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
                return y.half()
            elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
                return y.bfloat16()
            else:
                return y.float()

        @staticmethod
        def backward(ctx, gy):
            B = ctx.B
            T = ctx.T
            C = ctx.C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w, u, k, v, y = ctx.saved_tensors
            gw = torch.empty((B, C), device=gy.device,
                             memory_format=torch.contiguous_format)
            gu = torch.empty((B, C), device=gy.device,
                             memory_format=torch.contiguous_format)
            gk = torch.empty((B, T, C), device=gy.device,
                             memory_format=torch.contiguous_format)
            gv = torch.empty((B, T, C), device=gy.device,
                             memory_format=torch.contiguous_format)
            if '32' in os.environ['RWKV_FLOAT_MODE']:
                wkv_cuda.backward(B, T, C, w, u, k, v, y,
                                  gy.contiguous(), gw, gu, gk, gv)
            else:
                wkv_cuda.backward(B, T, C, w, u, k, v, y,
                                  gy.float().contiguous(), gw, gu, gk, gv)
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            if '32' in os.environ['RWKV_FLOAT_MODE']:
                return (None, None, None, gw, gu, gk, gv)
            elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
                return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
            elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
                return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
            else:
                return (None, None, None, gw.float(), gu.float(), gk.float(), gv.float())


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w, u, k, v)


"""
RWKV: RWKV Time-mix + RWKV Channel-mix
"""


class RWKV_TimeMix(BaseModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ctx_len = args.ctx_len
        self.n_embd = args.n_embd

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for h in range(args.dim_att):
                decay_speed[h] = -5 + 8 * \
                    (h / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)

            # fancy time_first
            zigzag = torch.tensor(
                [(i + 1) % 3 - 1 for i in range(args.dim_att)]) * 0.5
            self.time_first = nn.Parameter(torch.ones(
                args.dim_att) * math.log(0.3) + zigzag)

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(
                torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(
                torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        if hasattr(self.args, 'lora') and self.args.lora:
            self.key = make_linear_att(
                args.n_embd, args.dim_att, bias=False, lora_params=args.lora_params)
            self.value = make_linear_att(
                args.n_embd, args.dim_att, bias=False, lora_params=args.lora_params)
            self.receptance = make_linear_att(
                args.n_embd, args.dim_att, bias=False, lora_params=args.lora_params)
        else:
            self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)

    @JitFunction
    def jit_func(self, x):
        # Mix x with the previous timestep to produce xk, xv, xr
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        return sr, k, v

    def forward(self, x):
        B, T, C = x.size()  # x = (Batch,Time,Channel)
        sr, k, v = self.jit_func(x)
        rwkv = sr * RUN_CUDA(B, T, self.args.dim_att,
                             self.time_decay, self.time_first, k, v)
        return self.output(rwkv)


class RWKV_ChannelMix(BaseModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

        if hasattr(self.args, 'lora') and self.args.lora:
            self.key = make_linear_ffn(
                args.n_embd, args.dim_ffn, bias=False, lora_params=args.lora_params)
            self.receptance = make_linear_ffn(
                args.n_embd, args.n_embd, bias=False, lora_params=args.lora_params)
            self.value = make_linear_ffn(
                args.dim_ffn, args.n_embd, bias=False, lora_params=args.lora_params)
        else:
            self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @JitFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


"""
The RWKV Model with our blocks
"""


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if hasattr(args, 'dropout_p'):
            self.dropout = nn.Dropout(p=args.dropout_p)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
            if args.my_pos_emb > 0:
                self.pos_emb_x = nn.Parameter(
                    torch.zeros((1, args.my_pos_emb, args.n_embd)))
                self.pos_emb_y = nn.Parameter(
                    torch.zeros((args.my_pos_emb, 1, args.n_embd)))

        if self.layer_id == 0 and self.args.pre_ffn > 0:
            self.ffnPre = RWKV_ChannelMix(args, 0)
        else:
            self.att = RWKV_TimeMix(args, layer_id)

        self.ffn = RWKV_ChannelMix(args, layer_id)

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(args.n_embd)
            self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.register_buffer('tiny_mask', torch.tril(
                torch.ones(args.ctx_len, args.ctx_len)))

    def forward(self, x, x_emb=None):
        args = self.args
        has_dropout = hasattr(args, 'dropout_p')
        B, T, C = x.size()

        if self.layer_id == 0:
            x = self.ln0(x)
            if args.my_pos_emb > 0:
                pos_emb = (self.pos_emb_x +
                           self.pos_emb_y).reshape(T+1, -1)[:-1, :]
                x = x + pos_emb

        if self.layer_id == 0 and args.pre_ffn > 0:
            x = x + self.ffnPre(self.ln1(x))
        else:
            x = x + self.att(self.ln1(x)) if not has_dropout else \
                self.dropout(x + self.att(self.ln1(x)))

        x = x + self.ffn(self.ln2(x)) if not has_dropout else \
            self.dropout(x + self.ffn(self.ln2(x)))

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            xx = self.tiny_ln(x)
            q = self.tiny_q(xx)[:, :T, :]
            k = self.tiny_k(xx)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
            c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
            x = x + c @ self.tiny_v(x_emb)
        return x


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, 'dim_att'):
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn'):
            args.dim_ffn = args.n_embd * 4
        if not hasattr(args, 'tiny_att_layer'):
            args.tiny_att_layer = -1
        if not hasattr(args, 'tiny_att_dim'):
            args.tiny_att_dim = -1

        if args.vae_emb != None and args.vae_emb['enabled']:
            embed_dim = args.vae_emb['embed_dim']
            latent_dim = args.vae_emb['latent_dim']
            hidden_n = args.vae_emb['hidden_n']
            vocab_size = args.vae_emb['vocab_size']

            if args.vae_emb['base_model'] != None:
                logger.info(
                    f"Preloading embeddings from {args.vae_emb['base_model']}...")

                self.emb = VAE.from_pretrained(
                    args.vae_emb['base_model'],
                    embed_dim,
                    latent_dim,
                    hidden_n,
                    vocab_size
                )
            else:
                self.emb = VAE(
                    embed_dim,
                    latent_dim,
                    hidden_n,
                    vocab_size,
                )
        else:
            self.emb = nn.Embedding(
                args.vocab_size, args.n_embd, padding_idx=args.padding_idx)

        self.blocks = nn.ModuleList([Block(args, i)
                                    for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.head_qk > 0:
            self.head_q = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.head_k = nn.Linear(args.n_embd, args.head_qk, bias=False)

            self.register_buffer('copy_mask', torch.tril(
                torch.ones(args.ctx_len, args.ctx_len)))

    def configure_optimizers(self):
        args = self.args
        if args.layerwise_lr > 0:
            lr_1x = set()
            lr_2x = set()
            lr_3x = set()
            for n, p in self.named_parameters():
                if 'time_mix' in n:
                    if args.my_pile_stage == 2:
                        lr_2x.add(n)
                    else:
                        lr_1x.add(n)
                elif 'time_decay' in n:
                    if args.my_pile_stage == 2:
                        lr_3x.add(n)
                    else:
                        lr_2x.add(n)
                elif 'time_first' in n:
                    lr_3x.add(n)
                else:
                    lr_1x.add(n)
            lr_1x = sorted(list(lr_1x))
            lr_2x = sorted(list(lr_2x))
            lr_3x = sorted(list(lr_3x))

            param_dict = {n: p for n, p in self.named_parameters()}

            if args.my_pile_stage == 2:
                optim_groups = [
                    {'params': [param_dict[n] for n in lr_1x],
                        'weight_decay': 0.0, 'my_lr_scale': 1.0},
                    # test: 2e-3 / args.lr_init},
                    {'params': [param_dict[n] for n in lr_2x],
                        'weight_decay': 0.0, 'my_lr_scale': 5.0},
                    # test: 3e-3 / args.lr_init},
                    {'params': [param_dict[n] for n in lr_3x],
                        'weight_decay': 0.0, 'my_lr_scale': 5.0},
                ]
            else:
                optim_groups = [
                    {'params': [param_dict[n] for n in lr_1x],
                        'weight_decay': 0.0, 'my_lr_scale': 1.0},
                    {'params': [param_dict[n] for n in lr_2x],
                        'weight_decay': 0.0, 'my_lr_scale': 2.0},
                    {'params': [param_dict[n] for n in lr_3x],
                        'weight_decay': 0.0, 'my_lr_scale': 3.0},
                ]
        else:
            optim_groups = [
                {'params': [p for n, p in self.named_parameters()],
                 'weight_decay': 0.0},
            ]

        if hasattr(self.args, 'lora') and self.args.lora:
            for g in optim_groups:
                g["params"] = [p for p in g["params"] if p.requires_grad]

            optim_groups = [g for g in optim_groups if len(g["params"]) > 0]

        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0.01, amsgrad=False)

        return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0.01, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config['zero_optimization']
            return cfg.get('offload_optimizer') or cfg.get('offload_param')
        return False

    def forward(self, idx):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, 'Cannot forward, model ctx_len is exhausted.'

        if args.vae_emb != None and args.vae_emb['enabled']:
            if args.vae_emb['base_model'] != None:
                with torch.no_grad():
                    output, emb_hat, emb, hidden, mean, logvar = self.emb(idx)
            else:
                output, emb_hat, emb, hidden, mean, logvar = self.emb(idx)

            x = emb_hat

            self.register_buffer('emb_input', idx.detach().clone(), persistent=False)
            self.register_buffer('emb_output', output, persistent=False)
            self.register_buffer('emb_hat', emb_hat.detach().clone(), persistent=False)
            self.register_buffer('emb_orig', emb, persistent=False)
            self.register_buffer('emb_hidden', hidden, persistent=False)
            self.register_buffer('emb_mean', mean, persistent=False)
            self.register_buffer('emb_var', logvar, persistent=False)
        else:
            x = self.emb(idx)

        x_emb = x

        if args.tiny_att_dim > 0:
            for bb, block in enumerate(self.blocks):
                if args.grad_cp == 1:
                    if hasattr(self.args, 'lora') and self.args.lora:
                        x = torch_checkpoint(block, x, x_emb, use_reentrant=False)
                    else:
                        x = deepspeed.checkpointing.checkpoint(block, x, x_emb)
                else:
                    x = block(x, x_emb)
        else:
            for bb, block in enumerate(self.blocks):
                if args.grad_cp == 1:
                    if hasattr(self.args, 'lora') and self.args.lora:
                        x = torch_checkpoint(block, x, x_emb, use_reentrant=False)
                    else:
                        x = deepspeed.checkpointing.checkpoint(block, x)
                else:
                    x = block(x)

        x = self.ln_out(x)

        if args.head_qk > 0:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (1.0 / args.head_qk)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)

            if '32' in os.environ['RWKV_FLOAT_MODE']:
                c = c @ F.one_hot(idx, num_classes=args.vocab_size)
            elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
                c = c @ F.one_hot(idx, num_classes=args.vocab_size).half()
            elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
                c = c @ F.one_hot(idx, num_classes=args.vocab_size).bfloat16()
            else:
                c = c @ F.one_hot(idx, num_classes=args.vocab_size).float()

            x = self.head(x) + c
        else:
            x = self.head(x)

        return x

    def training_step(self, batch, batch_idx):
        args = self.args
        if args.my_qa_mask != 1:
            idx, targets = batch
            logits = self(idx)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            idx, targets, mask = batch
            mask = mask.view(-1)
            sum_mask = torch.sum(mask).item()
            # if sum_mask == 0:
            #     return torch.tensor([0.0], requires_grad=True)

            logits = self(idx)
            if sum_mask == mask.shape[0]:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
                # loss_raw = loss
                loss = torch.sum(loss * mask) / sum_mask

        if args.vae_emb != None and args.vae_emb['enabled']:
            if args.vae_emb['base_model'] is None:
                loss += (self.emb.loss_function(
                    self.emb_orig,
                    self.emb_hat,
                    self.emb_input,
                    self.emb_output,
                    self.emb_mean,
                    self.emb_var,
                    padding_index=0
                )) / 1000  # scale down

        return L2Wrap.apply(loss, logits)

    def on_train_batch_end(self,  outputs, batch, batch_idx):
        all = self.all_gather(outputs['loss'])
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all
