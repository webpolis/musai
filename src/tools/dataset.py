import json
import os
import numpy as np
from pathlib import Path
from typing import Any, Tuple, List
from torch import LongTensor, tensor, long, stack, bfloat16
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from pytorch_lightning.utilities import rank_zero_info
from binidx import MMapIndexedDataset
from utils import MaybeIsPrime


class MIDIDataset(Dataset):
    def __init__(self, files_paths: List[Path], min_seq_len: int, max_seq_len: int,
                 tokenizer=None, no_labels=False, batches=None, epoch_steps=None):
        self.no_labels = no_labels
        self.batches = batches
        self.epoch_steps = epoch_steps
        self.ctx_len = max_seq_len
        self.vocab_size = len(tokenizer)
        self.samples = []
        token_ids = []
        tokens = None

        for file_path in tqdm(files_paths, desc=f'Loading data: {files_paths[0].parent}'):
            with open(file_path) as json_file:
                try:
                    ids = json.load(json_file)['ids']
                    tokens = ids[0] if isinstance(ids[0], list) else ids
                    token_ids += tokens

                    i = 0
                    while i < len(tokens):
                        # Ensure there are enough tokens to create a sample of max_seq_len + 1
                        if i + max_seq_len + 1 > len(tokens):
                            break  # Not enough tokens left for a full sample

                        sample = LongTensor(tokens[i:i + max_seq_len + 1])
                        self.samples.append(sample)

                        # Move to the next non-overlapping position
                        i += max_seq_len + 1
                except:
                    pass

        self.data = token_ids
        self.data_size = len(self.data)
        # Pad samples to ensure uniform length (max_seq_len + 1)
        self.samples = self.pad_samples(self.samples, 0)

    def __getitem__(self, idx) -> Tuple[LongTensor, LongTensor]:
        i = np.random.randint(0, len(self.samples))
        dix = self.samples[i].clone().detach().long()
        x = dix[:-1]
        y = dix[1:]

        return x, y

    def __len__(self):
        return self.epoch_steps * self.batches

    def __repr__(self): return self.__str__()

    def __str__(self) -> str: return 'No data loaded' if len(
        self) == 0 else f'{len(self.data)} samples'

    def pad_samples(self, samples: List[LongTensor], pad_token: int) -> LongTensor:
        length_of_first = samples[0].size()

        # Check if padding is necessary.
        are_tensors_same_length = all(
            x.size() == length_of_first for x in samples)

        if are_tensors_same_length:
            return stack(samples, dim=0).long()

        # Creating the full tensor and filling it with our data.
        return pad_sequence(samples, batch_first=True, padding_value=pad_token).long()


class RegularDataset(Dataset):
    def __init__(self, args):
        self.args = args

        if args.data_type == "binidx":
            self.vocab_size = args.vocab_size
            rank_zero_info(
                f"Current vocab size = {self.vocab_size} (make sure it's correct)")

            if args.data_file.endswith('/'):
                d_all = []
                for p in os.listdir(args.data_file):
                    if p.endswith(".idx"):
                        d_all += [p[:-4]]
                d_all.sort()
                rank_zero_info(d_all)
                exit(0)
            else:
                self.data = MMapIndexedDataset(args.data_file)
                self.data_size = len(
                    self.data._bin_buffer) // self.data._index._dtype_size
                rank_zero_info(f"Data has {self.data_size} tokens.")
        elif args.data_type == "numpy":
            self.data = np.load(args.data_file).astype("int")
            self.vocab_size = args.vocab_size
            rank_zero_info("Current vocab size =", self.vocab_size,
                           "(make sure it's correct)")
            self.data_size = len(self.data)
            rank_zero_info(f"Data has {self.data_size} tokens.")
        elif args.data_type == "uint16":
            self.data = np.fromfile(args.data_file, dtype=np.uint16).astype(
                "int32").reshape(-1, args.my_sample_len)
            self.vocab_size = args.vocab_size
            rank_zero_info("Current vocab size =", self.vocab_size,
                           "(make sure it's correct)")
            self.data_size = self.data.shape[0]
            rank_zero_info(f"Data has {self.data_size} samples.")
        else:
            if args.data_type == "dummy":
                rank_zero_info("Building dummy data...")
                self.data = ""
                for i in range(100000):
                    aa = (i) % 10000
                    bb = (i * i) % 10000
                    cc = aa + bb
                    self.data += f".{aa}+{bb}={cc}."
            else:
                self.data = open(args.data_file, "r",
                                 encoding=args.data_type).read()

            rank_zero_info("Building token list...")

            unique = sorted(list(set(self.data)))
            self.vocab_size = len(unique)
            xx = 0
            xxObj = {}

            for u in unique:
                xxObj[xx] = u
                xx += 1

            with open(f"{args.proj_dir}/vocab.json", "w", encoding="utf-16le") as vocab_file:
                vocab_file.write(json.dumps(xxObj, ensure_ascii=False))

            self.data_size = len(self.data)

            rank_zero_info(
                f"Data has {self.data_size} tokens, {self.vocab_size} vocab size.")

            self.stoi = {ch: i for i, ch in enumerate(unique)}
            self.itos = {i: ch for i, ch in enumerate(unique)}

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args

        if args.data_type == "uint16":
            i = np.random.randint(0, self.data_size-1)
            dix = self.data[i]
            x = tensor(dix[:-1], dtype=long)
            y = tensor(dix[1:], dtype=long)
        else:
            ctx_len = args.ctx_len
            req_len = ctx_len + 1
            data = self.data

            # cheat: pick a random spot in dataset
            i = np.random.randint(0, self.data_size - req_len)

            if args.data_type == "binidx":
                dix = data.get(idx=0, offset=i, length=req_len).astype(int)
            elif args.data_type == "numpy":
                dix = data[i: i + req_len]
            else:
                dix = [self.stoi[s] for s in data[i: i + req_len]]

            x = tensor(dix[:-1], dtype=long)
            y = tensor(dix[1:], dtype=long)

            if args.my_qa_mask == 1:
                return x, y, z

            return x, y
