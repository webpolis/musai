import json
import numpy as np
from pathlib import Path
from typing import Any, Tuple, List
from miditok import MIDITokenizer
from torch import LongTensor, tensor, long, stack
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class MIDIDataset(Dataset):
    def __init__(self, files_paths: List[Path], min_seq_len: int, max_seq_len: int,
                 tokenizer: MIDITokenizer = None, no_labels=False, batches=None, epoch_steps=None):
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
                ids = json.load(json_file)['ids']
                tokens = ids[0] if isinstance(
                    ids[0], list) else ids  # first track (REMI, MMM)
                token_ids += tokens

                i = 0
                while i < len(tokens):
                    if i >= len(tokens) - min_seq_len:
                        break  # last sample is too short

                    sample = LongTensor(tokens[i:i + max_seq_len])

                    self.samples.append(sample)

                    i += len(self.samples[-1])  # could be replaced with self.ctx_len

        self.data = token_ids
        self.data_size = len(self.data)
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
        are_tensors_same_length = all(x.size() == length_of_first for x in samples)

        if are_tensors_same_length:
            return stack(samples, dim=0).long()

        # Creating the full tensor and filling it with our data.
        return pad_sequence(samples, batch_first=True, padding_value=pad_token).long()
