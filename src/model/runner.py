import types
import torch
import numpy as np
import os
import gc
from torch.nn import functional as F, Module
from typing import List, Dict

"""
Utility functions
"""


def __nop(ob):
    return ob


"""
Setup JIT
"""
BaseModule = Module
JitFunction = __nop

if 'RWKV_JIT_ON' in os.environ and os.environ['RWKV_JIT_ON'] == '1':
    BaseModule = torch.jit.ScriptModule
    JitFunction = torch.jit.script_method


DEBUG_TIME = False   # True False - show trained time-coeffs
RWKV_RESCALE_LAYER = 6  # set x=x/2 every X layer


class RWKV_RNN(BaseModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.FLOAT_MODE = args.FLOAT_MODE
        self.RUN_DEVICE = args.RUN_DEVICE

        with torch.no_grad():
            w = torch.load(args.base_model + '.pth',
                           map_location=args.map_location)

            if hasattr(self.args, 'lora'):
                # merge LoRA-only slim checkpoint into the main weights
                w_lora = torch.load(args.MODEL_LORA + '.pth', map_location='cpu')

                for k in w_lora.keys():
                    w[k] = w_lora[k]

                # merge LoRA weights
                keys = set(w.keys())

                for k in keys:
                    k: str
                    if k.endswith('.weight'):
                        prefix = k[:-len('.weight')]
                        lora_A = prefix + '.lora_A'
                        lora_B = prefix + '.lora_B'

                        if lora_A in keys:
                            assert lora_B in keys
                            print(f'merging {lora_A} and {lora_B} into {k}')
                            assert w[lora_B].shape[1] == w[lora_A].shape[0] == args.lora_r

                            # merging needs matmul, which is slow on cpu; work on gpu if possible
                            if args.RUN_DEVICE == 'cuda':
                                w[k] = w[k].cuda()
                                w[lora_A] = w[lora_A].cuda()
                                w[lora_B] = w[lora_B].cuda()
                            w[k] += w[lora_B] @ w[lora_A] * \
                                (args.lora_alpha / args.lora_r)

                            del w[lora_A]
                            del w[lora_B]

            # refine weights and send to correct device
            keys = list(w.keys())
            if 'pos_emb_x' in keys:
                w['pos_emb'] = (w['pos_emb_x'] + w['pos_emb_y']
                                ).reshape(args.ctx_len+1, -1)[:-1, :]
            keys = list(w.keys())
            print_need_newline = False
            for x in keys:
                block_id = 0
                if 'blocks.' in x:
                    block_id = int(x.split('.')[1])
                if 'att.output.weight' in x:
                    w[x] = w[x] / (2 ** int(block_id // RWKV_RESCALE_LAYER))
                if 'ffn.value.weight' in x:
                    w[x] = w[x] / (2 ** int(block_id // RWKV_RESCALE_LAYER))

                if '.time_' in x:
                    w[x] = w[x].squeeze()
                    if DEBUG_TIME:
                        print(x, w[x].numpy())
                if '.time_decay' in x:
                    w[x] = w[x].float()
                    w[x] = -torch.exp(w[x])
                elif '.time_first' in x:
                    w[x] = w[x].float()
                else:
                    if self.FLOAT_MODE == "fp32":
                        w[x] = w[x].float()
                    elif self.FLOAT_MODE == "bf16":
                        w[x] = w[x].bfloat16()
                    elif self.FLOAT_MODE == "fp16":
                        w[x] = w[x].half()

                w[x].requires_grad = False
                if args.RUN_DEVICE == 'cuda' and x != 'emb.weight':
                    w[x] = w[x].cuda()

                if ('blocks.' not in x) or ('blocks.0.' in x):
                    if print_need_newline:
                        print('\n', end='')
                        print_need_newline = False
                    print(x.ljust(40), str(w[x].dtype).replace(
                        'torch.', '').ljust(10), w[x].device)
                else:
                    print_need_newline = True
                    print('.', end='', flush=True)

        # store weights in self.w
        keys = list(w.keys())
        self.w = types.SimpleNamespace()
        for x in keys:
            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        self.eval()
        gc.collect()
        torch.cuda.empty_cache()

    def LN(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    # state[] 0=ffn_xx 1=att_xx 2=att_aa 3=att_bb 4=att_pp

    @JitFunction
    def FF(self, x, state, i: int, time_mix_k, time_mix_r, kw, vw, rw):
        if self.FLOAT_MODE == "bf16":
            xk = x * time_mix_k + \
                state[5*i+0].type(torch.bfloat16) * (1 - time_mix_k)
            xr = x * time_mix_r + \
                state[5*i+0].type(torch.bfloat16) * (1 - time_mix_r)
            state[5*i+0] = x.float()
        elif self.FLOAT_MODE == "fp16":
            xk = x * time_mix_k + state[5*i+0].half() * (1 - time_mix_k)
            xr = x * time_mix_r + state[5*i+0].half() * (1 - time_mix_r)
            state[5*i+0] = x.float()
        else:
            xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
            xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
            state[5*i+0] = x

        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk))
        kv = vw @ k

        return r * kv

    @JitFunction
    def SA(self, x, state, i: int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        if self.FLOAT_MODE == "bf16":
            xk = x * time_mix_k + \
                state[5*i+1].type(torch.bfloat16) * (1 - time_mix_k)
            xv = x * time_mix_v + \
                state[5*i+1].type(torch.bfloat16) * (1 - time_mix_v)
            xr = x * time_mix_r + \
                state[5*i+1].type(torch.bfloat16) * (1 - time_mix_r)
            state[5*i+1] = x.float()
        elif self.FLOAT_MODE == "fp16":
            xk = x * time_mix_k + state[5*i+1].half() * (1 - time_mix_k)
            xv = x * time_mix_v + state[5*i+1].half() * (1 - time_mix_v)
            xr = x * time_mix_r + state[5*i+1].half() * (1 - time_mix_r)
            state[5*i+1] = x.float()
        else:
            xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
            xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
            xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)
            state[5*i+1] = x

        r = torch.sigmoid(rw @ xr)
        k = kw @ xk
        v = vw @ xv

        if '16' in self.FLOAT_MODE:
            kk = k.float()
            vv = v.float()
        else:
            kk = k
            vv = v
        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + kk
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        a = e1 * aa + e2 * vv
        b = e1 * bb + e2
        ww = pp + time_decay
        p = torch.maximum(ww, kk)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kk - p)
        state[5*i+2] = e1 * aa + e2 * vv
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = p
        if self.FLOAT_MODE == "bf16":
            wkv = (a / b).type(torch.bfloat16)
        elif self.FLOAT_MODE == "fp16":
            wkv = (a / b).half()
        else:
            wkv = a / b

        return ow @ (r * wkv)

    def forward(self, ctx, state, preprocess_only=False):
        with torch.no_grad():
            w = self.w
            args = self.args

            x = w.emb.weight[ctx[-1]]
            if self.RUN_DEVICE == 'cuda':
                x = x.cuda()
            try:
                pos_emb = w.pos_emb[len(ctx)-1]
                x = x + pos_emb
            except:
                pass

            if state == None:
                state = torch.zeros(
                    args.n_layer * 5, args.n_embd, device=self.RUN_DEVICE)
                for i in range(args.n_layer):
                    state[5*i+4] -= 1e30

            for i in range(args.n_layer):
                if i == 0:
                    x = self.LN(x, w.blocks[i].ln0)

                ww = w.blocks[i].att
                x = x + self.SA(self.LN(x, w.blocks[i].ln1), state, i,
                                ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay,
                                ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight)

                ww = w.blocks[i].ffn
                x = x + self.FF(self.LN(x, w.blocks[i].ln2), state, i,
                                ww.time_mix_k, ww.time_mix_r,
                                ww.key.weight, ww.value.weight, ww.receptance.weight)

                if (i+1) % RWKV_RESCALE_LAYER == 0:
                    x = x / 2

            if preprocess_only:
                return state

            x = self.LN(x, w.ln_out)
            x = w.head.weight @ x

            return x.float(), state


def sample_logits(logits, temperature=1.0, top_p=0.85, top_k=0):
    probs = F.softmax(logits.float(), dim=-1)
    top_k = int(top_k)

    if probs.device == torch.device('cpu'):
        probs = probs.numpy()
        sorted_ids = np.argsort(probs)
        sorted_probs = probs[sorted_ids][::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        probs[probs < cutoff] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        probs = probs / np.sum(probs)
        out = np.random.choice(a=len(probs), p=probs)

        return int(out)
    else:
        sorted_ids = torch.argsort(probs)
        sorted_probs = probs[sorted_ids]
        sorted_probs = torch.flip(sorted_probs, dims=(0,))
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        probs[probs < cutoff] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        out = torch.multinomial(probs, num_samples=1)[0]

        return int(out)


def repetition_penalty(scores, history, ignore_tokens=[], repetition_penalty=1.1, max_penalty=1.5, seq_len=128, decay_factor=0.85):
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
