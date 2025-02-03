from tokenizers import Tokenizer
import types
import torch
import os
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
np.set_printoptions(precision=4, suppress=True, linewidth=200)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)

'''
This will load RWKV-7 "Goose" x070 and inference in GPT-mode (slower than RNN-mode for autoregressive generation)
'''

args = types.SimpleNamespace()

"""
Utility functions
"""


def __nop(ob):
    return ob


MODEL_PATH = ''
# MODEL_PATH = "/mnt/program/RWKV-x070-Pile-421M-20241127-ctx4096.pth"

if '168M' in MODEL_PATH:
    args.n_layer = 12
    args.n_embd = 768
    D_DECAY_LORA = 64
    D_AAA_LORA = 64
    D_MV_LORA = 32
    D_GATE_LORA = 128
elif '421M' in MODEL_PATH:
    args.n_layer = 24
    args.n_embd = 1024
    D_DECAY_LORA = 64
    D_AAA_LORA = 64
    D_MV_LORA = 64
    D_GATE_LORA = 128
else:
    D_DECAY_LORA = 64
    D_AAA_LORA = 64
    D_MV_LORA = 32
    D_GATE_LORA = 128

args.vocab_size = 50304  # "pile" model: 50277 padded to 50304

DTYPE = torch.bfloat16
# DTYPE = torch.half  # better

args.head_size_a = 64  # don't change
HEAD_SIZE = args.head_size_a

USE_CUDA_KERNEL = True  # False => UNOPTIMIZED, VERY SLOW

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script

########################################################################################################
# CUDA Kernel
########################################################################################################

if USE_CUDA_KERNEL:
    CUDA_SRC_PATH = f'{os.path.dirname(__file__)}/cuda'

    from torch.utils.cpp_extension import load

    load(name="wkv7", sources=[f"{CUDA_SRC_PATH}/wkv7_op_runner.cpp", f"{CUDA_SRC_PATH}/wkv7_runner.cu"], is_python_module=False,
         verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])

    class WKV_7(torch.autograd.Function):
        @staticmethod
        def forward(ctx, r, w, k, v, a, b):
            with torch.no_grad():
                B, T, C = r.size()
                H = C // HEAD_SIZE
                N = HEAD_SIZE
                assert HEAD_SIZE == C // H
                assert r.dtype == DTYPE
                assert w.dtype == DTYPE
                assert k.dtype == DTYPE
                assert v.dtype == DTYPE
                assert a.dtype == DTYPE
                assert b.dtype == DTYPE
                assert r.is_contiguous()
                assert w.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert a.is_contiguous()
                assert b.is_contiguous()
                y = torch.empty((B, T, C), device=k.device, dtype=DTYPE,
                                memory_format=torch.contiguous_format)
                torch.ops.wkv7.forward(B, T, C, H, r, w, k, v, a, b, y)
                return y

    def RWKV7_OP(r, w, k, v, a, b):
        return WKV_7.apply(r, w, k, v, a, b)

else:

    def RWKV7_OP(r, w, k, v, a, b):
        B, T, C = r.size()
        H = C // HEAD_SIZE
        N = HEAD_SIZE
        r = r.view(B, T, H, N).float()
        k = k.view(B, T, H, N).float()
        v = v.view(B, T, H, N).float()
        a = a.view(B, T, H, N).float()
        b = b.view(B, T, H, N).float()
        w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
        out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
        state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)

        for t in range(T):
            kk = k[:, t, :].view(B, H, 1, N)
            rr = r[:, t, :].view(B, H, N, 1)
            vv = v[:, t, :].view(B, H, N, 1)
            aa = a[:, t, :].view(B, H, N, 1)
            bb = b[:, t, :].view(B, H, 1, N)
            state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
            out[:, t, :] = (state @ rr).view(B, H, N)

            # another method using einsum
            #
            # kk = k[:, t, :]
            # rr = r[:, t, :]
            # vv = v[:, t, :]
            # aa = a[:, t, :]
            # bb = b[:, t, :]
            # sab = torch.einsum('bhik,bhk,bhj->bhij', state, aa, bb)
            # state = state * w[: , t, :, None, :] + sab + torch.einsum('bhj,bhi->bhij', kk, vv)
            # out[:, t, :] = torch.einsum('bhj,bhij->bhi', rr, state)

        return out.view(B, T, C).to(dtype=DTYPE)

########################################################################################################
# RWKV TimeMix
########################################################################################################


class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        H = self.n_head
        N = self.head_size
        C = args.n_embd

        self.x_r = nn.Parameter(torch.empty(1, 1, C))
        self.x_w = nn.Parameter(torch.empty(1, 1, C))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.x_v = nn.Parameter(torch.empty(1, 1, C))
        self.x_a = nn.Parameter(torch.empty(1, 1, C))
        self.x_g = nn.Parameter(torch.empty(1, 1, C))

        self.w0 = nn.Parameter(torch.empty(1, 1, C))
        self.w1 = nn.Parameter(torch.empty(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.empty(D_DECAY_LORA, C))

        self.a0 = nn.Parameter(torch.empty(1, 1, C))
        self.a1 = nn.Parameter(torch.empty(C, D_AAA_LORA))
        self.a2 = nn.Parameter(torch.empty(D_AAA_LORA, C))

        self.v0 = nn.Parameter(torch.empty(1, 1, C))
        self.v1 = nn.Parameter(torch.empty(C, D_MV_LORA))
        self.v2 = nn.Parameter(torch.empty(D_MV_LORA, C))

        self.g1 = nn.Parameter(torch.empty(C, D_GATE_LORA))
        self.g2 = nn.Parameter(torch.empty(D_GATE_LORA, C))

        self.k_k = nn.Parameter(torch.empty(1, 1, C))
        self.k_a = nn.Parameter(torch.empty(1, 1, C))
        self.r_k = nn.Parameter(torch.empty(H, N))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)  # !!! notice eps value !!!

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        # soft-clamp to (-inf, -0.5)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v  # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 +
                                                  # add value residual
                                                  (xv @ self.v1) @ self.v2)
        # a is "in-context learning rate"
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a-1) * self.k_a)

        x = RWKV7_OP(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B, T, H, -1)*k.view(B, T, H, -1)*self.r_k).sum(dim=-
                 1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x * g)
        return x, v_first

########################################################################################################
# RWKV ChannelMix
########################################################################################################


class RWKV_CMix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            self.x_k = nn.Parameter(torch.empty(1, 1, args.n_embd))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x

        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

########################################################################################################
# RWKV Block
########################################################################################################


class Block(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        # only used in block 0, should be fused with emb
        self.ln0 = nn.LayerNorm(args.n_embd)
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    @MyFunction
    def forward(self, x, v_first):

        if self.layer_id == 0:
            x = self.ln0(x)

        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx
        x = x + self.ffn(self.ln2(x))

        return x, v_first

########################################################################################################
# RWKV Model
########################################################################################################


class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.dim_att = args.n_embd
        args.dim_ffn = args.n_embd * 4
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i)
                                    for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, idx):

        x = self.emb(idx)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)

        x = self.ln_out(x)
        x = self.head(x)

        return x


def sample_logits(logits, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0):
    probs = F.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)

    if top_k > 0:
        probs[sorted_ids[top_k:]] = 0

    if top_p < 1:
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.searchsorted(cumulative_probs, top_p)
        cutoff = sorted_probs[cutoff_index]
        probs[probs < cutoff] = 0

        if top_p > 0:
            idx = torch.where(probs == cutoff)[0]
            if len(idx) > 0:
                probs[idx] = cutoff + \
                    (top_p - torch.sum(probs).item()) / len(idx)
                # assert abs(torch.sum(probs).item() - top_p) < 1e-6

    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)

    return torch.multinomial(probs, num_samples=1).item()


def repetition_penalty(scores, history, ignore_tokens=[], repetition_penalty=1.1, max_penalty=1.5, seq_len=128, decay_factor=0.85):
    """
    Applies a repetition penalty to the scores of generated tokens based on their repetition in the history.

    Args:
        scores (torch.Tensor): Tensor of scores for each generated token.
        history (List[int]): List of previously generated tokens.
        ignore_tokens (List[int], optional): List of tokens to ignore when calculating repetition penalties.
        repetition_penalty (float, optional): Penalty factor applied to repeated tokens. Higher values increase the penalty.
        max_penalty (float, optional): Maximum penalty factor that can be applied. This value restricts the maximum impact of the penalty.
        seq_len (int, optional): Length of the history or context considered for detecting repetitions. Default is 128.
        decay_factor (float, optional): Decay factor controlling the impact of token recency in the repetition penalty. A 
             value closer to 1 assigns more weight to recent tokens, while a value closer to 0 gives more weight to older tokens.

    Returns:
        torch.Tensor: Scores modified with the repetition penalty applied.
    """
    repetition_view_length = seq_len  # how far back to look for repetitions
    repetition_context = torch.tensor(
        history[-repetition_view_length:]).to(scores.device)

    # Exponential repetition penalty (accounts for recency and frequency)
    decays = torch.pow(torch.full_like(repetition_context, decay_factor, dtype=scores.dtype), torch.arange(
        0, repetition_context.shape[-1], device=scores.device, dtype=scores.dtype)).flip(-1)
    mask = torch.zeros_like(scores)
    mask.scatter_add_(-1, repetition_context, decays)
    mask[ignore_tokens] = 0
    penalty_factor = torch.pow(torch.where(
        scores < 0, repetition_penalty, 1 / repetition_penalty), mask)
    penalty_factor = torch.clamp(penalty_factor, torch.tensor(
        1.0 / max_penalty, device=penalty_factor.device), torch.tensor(max_penalty, device=penalty_factor.device))
    scores = scores * penalty_factor

    return scores
