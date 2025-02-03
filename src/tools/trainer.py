"""
MusAI

Author: Nicolás Iglesias
Email: nfiglesias@gmail.com

This file is part of MusAI, a project for generating MIDI music using
a combination of machine learning algorithms.

This training script is designed to train a model with various configurable options.
It utilizes command-line arguments to provide flexibility and control over the training
process.

MIT License
Copyright (c) [2023] [Nicolás Iglesias]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import argparse
import random
import numpy as np
import math
import time
import datetime
import subprocess
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import gc
import os
import torch
import pytorch_lightning as pl
import deepspeed
import sys
from multiprocessing import cpu_count
from loguru import logger
from pathlib import Path
from collections import namedtuple, OrderedDict
from typing import Dict
from lightning.pytorch.strategies import DeepSpeedStrategy
from torch.utils.data import DataLoader
from dataset import MIDIDataset, RegularDataset
from tokenizer import get_tokenizer, TOKEN_PARAMS_NAME

MODEL_SRC_PATH = f'{os.path.dirname(__file__)}/../model'
sys.path.append(MODEL_SRC_PATH)

"""
Some resets
"""

gc.collect()
torch.cuda.empty_cache()

"""
Some definitions
"""
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PRECISION = os.environ['PRECISION'] if 'PRECISION' in os.environ else 'bf16'
CTX_LEN = 1024

# training related
BATCHES = 5
N_EMBED = 768
N_LAYER = 24
DROPOUT = float(os.environ['DROPOUT']) if 'DROPOUT' in os.environ else 0.05
EPOCHS = 100
EPOCH_STEPS = 250
LR_RATE = 1e-4
LR_DECAY = 0

os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_FLOAT_MODE'] = 'fp16' if PRECISION == '16' else 'bf16'


def save_pth(dd, ff):
    torch.save(dd, ff)


class ResetValDataloader(pl.Callback):
    def on_validation_start(self, trainer, pl_module):
        trainer.reset_val_dataloader(pl_module)


class TrainCallback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.prefix = 'main'

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        # if args.cuda_cleanup > 0:
        #     torch.cuda.empty_cache()
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        # LR schedule
        w_step = args.warmup_steps
        if args.lr_final == args.lr_init or args.epoch_count == 0:
            lr = args.lr_init
        else:
            decay_step = real_step - args.my_pile_edecay * args.epoch_steps
            decay_total = (args.epoch_count -
                           args.my_pile_edecay) * args.epoch_steps
            progress = (decay_step - w_step + 1) / (decay_total - w_step)
            progress = min(1, max(0, progress))

            if args.lr_final == 0 or args.lr_init == 0:  # linear decay
                lr = args.lr_init + (args.lr_final - args.lr_init) * progress
            else:  # exp decay
                lr = args.lr_init * \
                    math.exp(math.log(args.lr_final / args.lr_init)
                             * pow(progress, 1))
            # if trainer.is_global_zero:
            #     print(trainer.global_step, decay_step, decay_total, w_step, progress, lr)

        if args.my_exit_tokens != 0:  # cosine decay
            real_tokens = real_step * args.ctx_len * args.real_bsz
            warmup_tokens = w_step * args.ctx_len * args.real_bsz
            progress = (real_tokens - warmup_tokens) / \
                (abs(args.my_exit_tokens) - warmup_tokens)
            progress = max(0, min(1, progress))
            lr_final_factor = args.lr_final / args.lr_init
            lr_mult = (0.5 + lr_final_factor / 2) + (0.5 -
                                                     lr_final_factor / 2) * math.cos(math.pi * progress)
            if args.my_exit_tokens > 0:
                lr = args.lr_init * lr_mult
            else:
                lr = (lr + args.lr_init * lr_mult) / 2
            if progress >= 1:
                if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):
                    to_save_dict = pl_module.state_dict()
                    save_pth(
                        to_save_dict,
                        f'{args.proj_dir}/{self.prefix}_{args.epoch_begin + trainer.current_epoch}.pth',
                    )
                    exit(0)
        if trainer.global_step < w_step:
            lr = lr * (0.01 + 0.99 * trainer.global_step / w_step)

        if args.weight_decay_final > 0:
            wd_now = args.weight_decay * \
                math.exp(math.log(args.weight_decay_final /
                         args.weight_decay) * progress)
        else:
            wd_now = args.weight_decay

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            if args.layerwise_lr > 0:
                param_group["lr"] = lr * param_group["my_lr_scale"]
                # print(param_group["lr"], param_group["my_lr_scale"])
            else:
                param_group["lr"] = lr

        trainer.my_lr = lr
        trainer.my_wd = wd_now
        # rank_zero_info(f"{real_step} {lr}")

        if trainer.global_step == 0:
            if trainer.is_global_zero:  # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
                try:
                    print(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except:
                    pass
                trainer.my_log.flush()
                if len(args.wandb) > 0:
                    print("Login to wandb...")
                    import wandb
                    wandb.init(
                        project=args.wandb,
                        name=args.run_name + " " + args.my_timestamp,
                        config=args,
                        save_code=False,
                    )
                    trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                self.log("REAL it/s", 1.0 / t_cost,
                         prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            if pl.__version__[0] == '2':
                trainer.my_loss = outputs["loss"]
            else:
                trainer.my_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_epoch_loss,
                     prog_bar=True, on_step=True)
            # self.log("s", real_step, prog_bar=True, on_step=True)

            if len(args.wandb) > 0:
                lll = {"loss": trainer.my_loss, "lr": trainer.my_lr,
                       "wd": trainer.my_wd, "Gtokens": real_step * token_per_step / 1e9}
                if kt_s > 0:
                    lll["kt/s"] = kt_s
                trainer.my_wandb.log(lll, step=int(real_step))
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):  # save pth
            if args.magic_prime > 0:
                expand_factor = 2 if args.my_qa_mask > 0 else 1
                if int(real_step) == int(args.magic_prime * expand_factor // args.real_bsz) - 1 + int(args.my_random_steps):
                    to_save_dict = pl_module.state_dict()
                    save_pth(
                        to_save_dict,
                        f'{args.proj_dir}/{self.prefix}_{args.epoch_begin + trainer.current_epoch}.pth',
                    )

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        if pl.__version__[0] == '2':
            dataset = trainer.train_dataloader.dataset
        else:
            dataset = trainer.train_dataloader.dataset.datasets
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size
        # print(f'########## world_size {dataset.world_size} global_rank {dataset.global_rank} real_epoch {dataset.real_epoch} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        to_save_dict = {}
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):  # save pth
            if (args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0) or (trainer.current_epoch == args.epoch_count - 1):
                if args.data_type == 'wds_img':
                    raw_dict = pl_module.state_dict()
                    for k in raw_dict:
                        if k.startswith('encoder.') or k.startswith('decoder.'):
                            to_save_dict[k] = raw_dict[k]
                else:
                    to_save_dict = pl_module.state_dict()
                try:
                    save_pth(
                        to_save_dict,
                        f'{args.proj_dir}/{self.prefix}_{args.epoch_begin + trainer.current_epoch}.pth',
                    )
                except Exception as e:
                    print('Error\n\n', e, '\n\n')

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(
                trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n")
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0
            if (args.epoch_begin + trainer.current_epoch) >= args.my_exit:
                exit(0)


"""
CLI
"""
if __name__ == "__main__":
    # parse command line arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-t', '--dataset_path', default=None,
                            help='The path were tokens parameters were saved by the tokenizer', type=str)
    arg_parser.add_argument('-x', '--binidx', help='Dataset is in binidx format',
                            action='store_true', default=False)
    arg_parser.add_argument('-o', '--output_path', default='out',
                            help='The output path were model binaries will be saved', type=str)
    arg_parser.add_argument('-m', '--base_model', default=None,
                            help='Full path for base model/checkpoint', type=str)
    arg_parser.add_argument('-r', '--lora_ckpt', default=None,
                            help='Full path for LoRa checkpoint', type=str)
    arg_parser.add_argument('-v', '--vae_emb', default=None, nargs='*',
                            help='The pre-trained VAE embeddings. Possible options: "train" for training alone, \
                                from scratch. "true" for training from scratch together with the main model (slow). \
                                    "path_to_pretrained_embeddings.pth" to use existing embeddings model (faster).', type=str)
    arg_parser.add_argument(
        '-c', '--ctx_len', default=CTX_LEN, help='The context length', type=int)
    arg_parser.add_argument(
        '-b', '--batches_num', default=BATCHES, help='Number of batches', type=int)
    arg_parser.add_argument(
        '-e', '--embed_num', default=N_EMBED, help='Size of the embeddings dimension', type=int)
    arg_parser.add_argument(
        '-n', '--layers_num', default=N_LAYER, help='Number of block layers', type=int)
    arg_parser.add_argument(
        '-f', '--epochs_first', default=0, help='Initial epoch', type=int)
    arg_parser.add_argument(
        '-p', '--epochs_num', default=EPOCHS, help='Number of epochs', type=int)
    arg_parser.add_argument(
        '-s', '--steps_num', default=EPOCH_STEPS, help='Number of steps per epoch', type=int)
    arg_parser.add_argument(
        '-i', '--lr_rate', default=str(LR_RATE), help='Learning rate. Initial & final derivates from it.', type=str)
    arg_parser.add_argument(
        '-d', '--lr_decay', default=str(LR_DECAY), help='Learning rate decay thru steps', type=str)
    arg_parser.add_argument('-a', '--attention', help='Enable tiny attention',
                            action='store_true', default=False)
    arg_parser.add_argument('-l', '--lora', help='Activate LoRa (Low-Rank Adaptation)',
                            action='store_true', default=False)
    arg_parser.add_argument('-u', '--offload', help='DeepSpeed offload',
                            action='store_true', default=False)
    arg_parser.add_argument('-q', '--head_qk', help='Enable head QK',
                            action='store_true', default=False)
    args = arg_parser.parse_args()

    if args.dataset_path == None:
        raise 'Invalid dataset path'

    os.environ['RWKV_T_MAX'] = str(args.ctx_len)

    from model import RWKV, LORA_CONFIG
    from embed import VAE, HIDDEN_N, LATENT_DIM

    # seed
    seed = random.randint(1000, 10000)
    pl.seed_everything(seed)

    # generate output dir
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    logger.info('Output dir setup.')

    # construct dataset
    logger.info('Initializing dataset...')

    if not args.binidx:
        midi_jsons = list(Path(args.dataset_path).glob('*.json'))
        random.shuffle(midi_jsons)

        # initialize tokenizer
        TOKENIZER = get_tokenizer(
            params=f'{args.dataset_path}/{TOKEN_PARAMS_NAME}')
        vocab_size = len(TOKENIZER)
    else:
        vocab_size = 65536

    # parse VAE params
    VAE_MODE = args.vae_emb[0] if args.vae_emb != None else None
    VAE_FILE = (
        VAE_MODE if (
            VAE_MODE != None and VAE_MODE != 'true' and VAE_MODE != 'train'
            and os.path.isfile(VAE_MODE)
        ) else
        args.vae_emb[1] if
        (
            args.vae_emb != None and
            len(args.vae_emb) > 1 and VAE_MODE == 'train' and os.path.isfile(
                args.vae_emb[1])
        ) else
        None
    )

    # build trainer/model params
    params = {
        'adam_eps': 1e-8,
        'betas': (.9, .99),
        'ctx_len': args.ctx_len,
        'dim_att': args.embed_num,
        'dim_ffn': args.embed_num * 4,
        'dropout': 0,
        'dropout_p': DROPOUT,
        'epoch_begin': args.epochs_first,
        'epoch_count': args.epochs_num,
        'epoch_save': 1,
        'epoch_steps': args.steps_num,
        'grad_cp': 0 if not args.offload else 1,
        'gradient_clip_val': 1.0,
        'head_qk': 0 if not args.head_qk else int(args.embed_num * 2),
        'layerwise_lr': 0,
        'lora': args.lora,
        'lora_params': LORA_CONFIG,
        'lr_decay': float(args.lr_decay),
        'lr_init': float(args.lr_rate),
        'lr_final': float(args.lr_rate) / 100,
        'micro_bsz': args.batches_num,
        'my_pile_stage': 0,
        'my_pos_emb': 0,
        'my_qa_mask': 0,
        'my_timestamp': datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S"),
        'n_embd': args.embed_num,
        'n_layer': args.layers_num,
        'padding_idx': 0,
        'pre_ffn': 0,
        'head_size_a': 64,
        'proj_dir': args.output_path,
        'real_bsz': args.batches_num,
        'strategy': 'deepspeed_stage_2_offload',
        'tiny_att_dim': -1 if not args.attention else args.ctx_len,
        'tiny_att_layer': -1 if not args.attention else int(args.layers_num) - 1,
        'vae_emb': {
            'enabled': VAE_MODE is not None,
            'training': VAE_MODE == 'train',
            'embed_dim': args.embed_num,
            'hidden_n': HIDDEN_N,
            'latent_dim': LATENT_DIM,
            'base_model': VAE_FILE,
        },
        'vocab_size': vocab_size,
        'wandb': '',
        'warmup_steps': int(args.steps_num * 0.01),
        'load_model': '',
        'random_seed': -1,
        'train_type': '',
        'data_file': '',
        'data_type': 'utf-8',
        'weight_decay': 0,
        'weight_decay_final': -1,
        'beta1': 0.9,
        'beta2': 0.99,
        'my_pile_version': 1,
        'my_pile_shift': -1,
        'my_pile_edecay': 0,
        'ds_bucket_mb': 200,
        'my_sample_len': 0,
        'my_ffn_shift': 1,
        'my_att_shift': 1,
        'head_size_divisor': 8,
        'load_partial': 0,
        'magic_prime': 0,
        'my_random_steps': 0,
        'my_testing': 'x052',
        'my_exit': 99999999,
        'my_exit_tokens': 0,
    }

    logger.info(params)

    # instantiate dataset
    if not args.binidx:
        DATASET = MIDIDataset(
            files_paths=midi_jsons,
            min_seq_len=16,
            max_seq_len=args.ctx_len,
            no_labels=False,
            tokenizer=TOKENIZER,
            batches=args.batches_num,
            epoch_steps=args.steps_num
        )
        params_obj = namedtuple('RWKVParams', params.keys())(*params.values())
    else:
        params['data_type'] = 'binidx'
        params['data_file'] = args.dataset_path
        params_obj = namedtuple('RWKVParams', params.keys())(*params.values())
        DATASET = RegularDataset(params_obj)

    # extra embeddings metadata
    if not args.binidx:
        params_obj.vae_emb['vocab_size'] = len(TOKENIZER)

    logger.info('Loading data...')
    data_loader = DataLoader(DATASET, shuffle=False, pin_memory=True,
                             batch_size=params_obj.micro_bsz, num_workers=cpu_count(), persistent_workers=False, drop_last=True)
    TRAIN_CALLBACK = TrainCallback(params_obj)

    try:
        # prepare for training
        trainer_params = {
            'gradient_clip_val': 1.0,
            'log_every_n_steps': args.steps_num//10,
            'devices': 'auto',
            'max_steps': args.steps_num*args.epochs_num,
            'accelerator': 'gpu',
            'enable_checkpointing': False,
            'precision': PRECISION,
            'callbacks': [TRAIN_CALLBACK],
        }

        DEEPSPEED_CONFIG = {
            # 'optimizer': {
            #     'type': 'Adam',
            #     'params': {
            #         'lr': params_obj.lr_init,
            #         'betas': params_obj.betas,
            #         'eps': params_obj.adam_eps,
            #         'weight_decay': 3e-7
            #     }
            # },
            # 'scheduler': {
            #     'type': 'WarmupDecayLR',
            #     'params': {
            #         'total_num_steps': params_obj.epoch_steps*params_obj.epoch_count,
            #         'warmup_min_lr': params_obj.lr_final,
            #         'warmup_max_lr': params_obj.lr_init,
            #         'warmup_num_steps': params_obj.warmup_steps
            #     }
            # },
            'bf16': {
                'enabled': PRECISION == 'bf16',
            },
            'fp16': {
                'enabled': PRECISION == '16',
            },
            'train_batch_size': args.batches_num,
            'train_micro_batch_size_per_gpu': args.batches_num
        }

        if args.offload:
            DEEPSPEED_CONFIG = {
                'zero_optimization': {
                    'stage': 2,
                    'allgather_partitions': False,
                    'allgather_bucket_size': 200 * 1000 * 1000,
                    'reduce_scatter': False,
                    'reduce_bucket_size': 200 * 1000 * 1000,
                    'overlap_comm': False,
                    'contiguous_gradients': False,
                    'offload_optimizer': {
                        'device': 'cpu'
                    },
                    'offload_param': {
                        'device': 'cpu',
                        'pin_memory': True
                    },
                }
            }

        # trainer_params['strategy'] = DeepSpeedStrategy(config=DEEPSPEED_CONFIG)
        trainer_params['strategy'] = 'deepspeed'
        trainer_pl = pl.Trainer(**trainer_params)

        if params_obj.vae_emb['enabled'] and params_obj.vae_emb['training']:
            # train the VAE model (embeddings) alone
            logger.info('Setting up trainer for embeddings model...')

            if VAE_FILE != None:
                logger.info(f'Preloading embeddings from {VAE_FILE}...')

                emb_model = VAE.from_pretrained(
                    VAE_FILE,
                    params_obj.vae_emb['embed_dim'],
                    params_obj.vae_emb['latent_dim'],
                    params_obj.vae_emb['hidden_n'],
                    params_obj.vae_emb['vocab_size'],
                )
            else:
                emb_model = VAE(
                    params_obj.vae_emb['embed_dim'],
                    params_obj.vae_emb['latent_dim'],
                    params_obj.vae_emb['hidden_n'],
                    params_obj.vae_emb['vocab_size'],
                )

            # begin training
            logger.info('Begin training the embeddings model...')
            trainer_pl.fit(emb_model, data_loader)
        else:
            # train the main model
            if args.lora and args.grad_cp:
                logger.info(
                    'LoRA Warning: Gradient Checkpointing requires JIT off, disabling it')
                os.environ["RWKV_JIT_ON"] = "0"
            model_base = RWKV(params_obj)
            logger.info('Setting up trainer for main model...')

            # LoRa customization
            if params_obj.lora:
                logger.info('LoRa enabled: preparing modules...')

                enable_time_finetune = 'time' in params_obj.lora_params['parts']
                enable_ln_finetune = 'ln' in params_obj.lora_params['parts']

                model_base.requires_grad_(False)

                for name, module in model_base.named_modules():
                    # have to check param name since it may have been wrapped by torchscript
                    if any(n.startswith("lora_") for n, _ in module.named_parameters()):
                        logger.debug(f'LoRA training module {name}')
                        for pname, param in module.named_parameters():
                            param.requires_grad = 'lora_' in pname
                    elif enable_ln_finetune and '.ln' in name:
                        logger.debug(
                            f'LoRA additionally training module {name}')
                        for param in module.parameters():
                            param.requires_grad = True
                    elif enable_time_finetune and any(n.startswith("time") for n, _ in module.named_parameters()):
                        for pname, param in module.named_parameters():
                            if pname.startswith("time"):
                                logger.debug(
                                    f'LoRA additionally training parameter {pname}')
                                param.requires_grad = True

            # Checkpoint preload
            if args.base_model != None and os.path.isfile(args.base_model):
                try:
                    logger.info(f'Preloading {args.base_model}')
                    load_dict = torch.load(args.base_model, map_location='cpu')
                    load_keys = load_dict.keys()

                    for k in model_base.state_dict():
                        if k not in load_keys:
                            load_dict[k] = model_base.state_dict()[k]

                    # If using LoRA, the LoRA keys might be missing in the original model
                    model_base.load_state_dict(
                        load_dict, strict=(not args.lora))

                    if args.lora and args.lora_ckpt != None \
                            and os.path.isfile(args.lora_ckpt):
                        logger.info(f'Preloading LoRa checkpoint {
                                    args.lora_ckpt}')

                        model_base.load_state_dict(torch.load(
                            args.lora_ckpt, map_location='cpu'), strict=False)
                except Exception as error:
                    logger.error(error)

            logger.info('Model initialized')

            # begin training
            logger.info('Begin training the main model...')
            trainer_pl.fit(model_base, data_loader)
    except KeyboardInterrupt:
        sys.exit(1)
