import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List
from miditok import MIDITokenizer
from torch import LongTensor, tensor, long
from torch.utils.data import Dataset
from tqdm import tqdm


class MIDIDataset(Dataset):
    def __init__(self, files_paths: List[Path], min_seq_len: int, max_seq_len: int,
                 tokenizer: MIDITokenizer = None, no_labels=False, batches=None, epoch_steps=None):
        self.no_labels = no_labels
        self.batches = batches
        self.epoch_steps = epoch_steps
        self.ctx_len = max_seq_len
        self.vocab_size = len(tokenizer)
        token_ids = []
        tokens = None

        for file_path in tqdm(files_paths, desc=f'Loading data: {files_paths[0].parent}'):
            with open(file_path) as json_file:
                ids = json.load(json_file)['ids']
                tokens = ids[0] if isinstance(
                    ids[0], list) else ids  # first track (REMI, MMM)
                token_ids += tokens

        self.data = token_ids
        self.data_size = len(self.data)

    def __getitem__(self, idx) -> Dict[str, LongTensor]:
        req_len = self.ctx_len + 1
        data = self.data
        i = np.random.randint(0, self.data_size - req_len)
        dix = data[i: i + req_len]
        x = tensor(dix[:-1], dtype=long)
        y = tensor(dix[1:], dtype=long)

        return x, y

    def __len__(self):
        return self.epoch_steps * self.batches

    def __repr__(self): return self.__str__()

    def __str__(self) -> str: return 'No data loaded' if len(
        self) == 0 else f'{len(self.data)} samples'
