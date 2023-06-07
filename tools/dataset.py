import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List
from miditok import MIDITokenizer
from torch import LongTensor, tensor, long
from torch.utils.data import Dataset
from tqdm import tqdm


class MIDIDataset(Dataset):
    def __init__(self, files_paths: List[Path], min_seq_len: int, max_seq_len: int, tokenizer: MIDITokenizer = None, no_labels=False):
        samples = []
        tokens = None
        self.no_labels = no_labels

        for file_path in tqdm(files_paths, desc=f'Loading data: {files_paths[0].parent}'):
            with open(file_path) as json_file:
                ids = json.load(json_file)['ids']
                tokens = ids[0] if isinstance(
                    ids[0], list) else ids  # first track (REMI, MMM)
            i = 0
            while i < len(tokens):
                if i >= len(tokens) - min_seq_len:
                    break  # last sample is too short
                samples.append(LongTensor(tokens[i:i + max_seq_len]))
                i += len(samples[-1])  # could be replaced with max_seq_len

        self.data = samples
        self.ctx_len = max_seq_len
        self.vocab_size = len(tokenizer.vocab)
        self.data_size = len(self.data)

    def __getitem__(self, idx) -> Dict[str, LongTensor]:
        req_len = self.ctx_len + 1
        data = self.data

        # cheat: pick a random spot in dataset
        i = np.random.randint(0, self.data_size - req_len)
        dix = data[i: i + req_len]

        x = tensor(dix[:-1], dtype=long)
        y = tensor(dix[1:], dtype=long)

        return x, y

    def __len__(self) -> int: return len(self.data)

    def __repr__(self): return self.__str__()

    def __str__(self) -> str: return 'No data loaded' if len(
        self) == 0 else f'{len(self.data)} samples'
