"""
MusAI

Author: Nicolás Iglesias
Email: nfiglesias@gmail.com

This file is part of MusAI, a project for generating MIDI music using 
a combination of machine learning algorithms.

Below is a monolitic script that performs the training over an already 
generated dataset of tokens (via the tokenizer.py script) and saves the 
model in a defined output directory.

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
import math
import time
import datetime
import argparse
import random
import gc
import os
import torch
import lightning.pytorch as pl
import sys
from multiprocessing import cpu_count
from loguru import logger
from pathlib import Path
from collections import namedtuple
from torchtoolkit.data import create_subsets
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
from dataset import MIDIDataset
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
PRECISION = 'bf16'
CTX_LEN = 2048

# training related
BATCHES = 5
N_EMBED = 768
N_LAYER = 10
EPOCHS = 100
EPOCH_STEPS = 250
LR_RATE = 1.5e-5
LR_DECAY = 5e-6

os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_FLOAT_MODE'] = PRECISION


def save_pth(dd, ff):
    torch.save(dd, ff)


class ResetValDataloader(Callback):
    def on_validation_start(self, trainer, pl_module):
        trainer.reset_val_dataloader(pl_module)


class TrainCallback(Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        # LR schedule
        w_step = args.warmup_steps

        if args.lr_final == args.lr_init or args.epoch_count == 0:
            lr = args.lr_init
            if trainer.global_step < w_step:
                lr = lr * (0.2 + 0.8 * trainer.global_step / w_step)
        else:
            decay_step = real_step - args.lr_decay * args.epoch_steps
            decay_total = (args.epoch_count -
                           args.lr_decay) * args.epoch_steps
            progress = (decay_step - w_step + 1) / (decay_total - w_step)
            progress = min(1, max(0, progress))

            if args.lr_final == 0 or args.lr_init == 0:  # linear decay
                lr = args.lr_init + (args.lr_final - args.lr_init) * progress
            else:  # exp decay
                lr = args.lr_init * \
                    math.exp(math.log(args.lr_final / args.lr_init)
                             * pow(progress, 1))

            if trainer.global_step < w_step:
                lr = lr * (0.2 + 0.8 * trainer.global_step / w_step)

        for param_group in trainer.optimizers[0].param_groups:
            if args.layerwise_lr > 0:
                param_group["lr"] = lr * param_group["my_lr_scale"]
            else:
                param_group["lr"] = lr

        trainer.my_lr = lr

        if trainer.global_step == 0:
            if trainer.is_global_zero:  # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
                trainer.my_log.write(
                    f"NEW RUN {args.my_timestamp}\n{self.args._asdict()}\n")

                try:
                    print(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except:
                    pass

                trainer.my_log.flush()

                if len(args.wandb) > 0:
                    logger.info("Login to wandb...")
                    import wandb

                    wandb.init(
                        project=args.wandb,
                        name=str(args.vocab_size) + "_" +
                        str(args.n_layer) + "_" + args.my_timestamp,
                        config=args._asdict(),
                        save_code=False,
                    )
                    trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args

        if trainer.is_global_zero:  # logging
            if not hasattr(trainer, 'my_loss_all'):
                trainer.my_loss_all = pl_module.all_gather(outputs['loss'])

            t_now = time.time_ns()
            token_per_step = args.ctx_len * args.real_bsz
            real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
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
            trainer.my_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count

            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_epoch_loss,
                     prog_bar=True, on_step=True)

            if len(args.wandb) > 0:
                lll = {"loss": trainer.my_loss, "lr": trainer.my_lr,
                       "Gtokens": real_step * token_per_step / 1e9}
                if kt_s > 0:
                    lll["kt/s"] = kt_s
                trainer.my_wandb.log(lll, step=int(real_step))

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        dataset = trainer.train_dataloader.dataset
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args

        if trainer.is_global_zero:  # logging & save state_dict
            if (args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0) or (trainer.current_epoch == args.epoch_count - 1):
                to_save_dict = pl_module.state_dict()

                try:
                    save_pth(
                        to_save_dict,
                        f"{args.proj_dir}/rwkv-{args.epoch_begin + trainer.current_epoch}.pth",
                    )
                except Exception as e:
                    print('Error\n\n', e, '\n\n')

            trainer.my_log.write(
                f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n")
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0


"""
CLI
"""
if __name__ == "__main__":
    # parse command line arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-t', '--tokens_path', default=None,
                            help='The path were tokens parameters were saved by the tokenizer', type=str)
    arg_parser.add_argument('-o', '--output_path', default='out',
                            help='The output path were model binaries will be saved', type=str)
    arg_parser.add_argument(
        '-c', '--ctx_len', default=CTX_LEN, help='The context length', type=int)
    arg_parser.add_argument(
        '-b', '--batches_num', default=BATCHES, help='Number of batches', type=int)
    arg_parser.add_argument(
        '-e', '--embed_num', default=N_EMBED, help='Size of the embeddings dimension', type=int)
    arg_parser.add_argument(
        '-l', '--layers_num', default=N_LAYER, help='Number of block layers', type=int)
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
    args = arg_parser.parse_args()

    if args.tokens_path == None:
        raise 'Invalid tokens path'

    try:
        # seed
        seed = random.randint(1000, 10000)
        pl.seed_everything(seed)

        # initialize tokenizer
        TOKENIZER = get_tokenizer(
            params=f'{args.tokens_path}/{TOKEN_PARAMS_NAME}')

        # generate output dir
        Path(args.output_path).mkdir(parents=True, exist_ok=True)
        logger.info('Output dir setup.')

        # construct dataset
        logger.info('Initializing dataset...')

        midi_jsons = list(Path(args.tokens_path).glob('*.json'))
        random.shuffle(midi_jsons)

        DATASET = MIDIDataset(
            files_paths=midi_jsons,
            min_seq_len=16,
            max_seq_len=args.ctx_len,
            no_labels=False,
            tokenizer=TOKENIZER,
            batches=args.batches_num,
            epoch_steps=args.steps_num
        )
        subset_train, subset_valid = create_subsets(DATASET, [0.3])

        # build trainer/model params
        logger.info('Setting up trainer...')

        params = {
            'adam_eps': 1e-8,
            'betas': (.9, .99),
            'ctx_len': args.ctx_len,
            'dim_att': args.embed_num,
            'dim_ffn': args.embed_num*4,
            'dropout_p': 0.25,
            'eight_bits': False,  # experimental
            'epoch_begin': 0,
            'epoch_count': args.epochs_num,
            'epoch_save': 2,
            'epoch_steps': args.steps_num,
            'grad_cp': 0,  # model.py:530
            'gradient_clip_val': 1.0,
            'head_qk': int(args.embed_num*2),
            'layerwise_lr': 1,
            'lr_decay': float(args.lr_decay),
            'lr_init': float(args.lr_rate),
            'lr_final': float(args.lr_rate)/100,
            'micro_bsz': args.batches_num,
            'my_pile_stage': 0,
            'my_pos_emb': 0,
            'my_qa_mask': 0,
            'my_timestamp': datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S"),
            'n_embd': args.embed_num,
            'n_layer': args.layers_num,
            'padding_idx': 0,
            'pre_ffn': 0,
            'proj_dir': args.output_path,
            'real_bsz':  args.batches_num,
            'strategy': 'ddp_find_unused_parameters_false',
            'tiny_att_dim': -1 if not args.attention else args.ctx_len,
            'tiny_att_layer': -1 if not args.attention else int(args.layers_num) - 1,
            'vocab_size': len(TOKENIZER),
            'wandb': 'musai',
            'warmup_steps': 10,
        }

        logger.info(params)

        params_obj = namedtuple('RWKVParams', params.keys())(*params.values())

        # initialize model
        os.environ['RWKV_T_MAX'] = str(args.ctx_len)
        from model import RWKV

        model_base = RWKV(params_obj)
        model_base.to(DEVICE)

        logger.info('Model initialized')

        # prepare for training
        logger.info('Loading data...')
        data_loader = DataLoader(DATASET, shuffle=False, pin_memory=True,
                                 batch_size=params_obj.micro_bsz, num_workers=cpu_count(), persistent_workers=False, drop_last=True)
        trainer_params = {
            'gradient_clip_val': 1.0,
            'log_every_n_steps': 100,
            'devices': 'auto',
            'max_steps': args.steps_num*args.epochs_num,
            'accelerator': 'gpu',
            'strategy': 'auto',
            'enable_checkpointing': True,
            'precision': '16',
            'callbacks': [TrainCallback(params_obj)],
        }
        trainer_pl = pl.Trainer(**trainer_params)

        # begin training
        logger.info('Begin training...')
        trainer_pl.fit(model_base, data_loader)
    except KeyboardInterrupt:
        sys.exit(1)
