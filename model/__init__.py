import os
import math
import gc
import importlib
import torch
import torch.nn as nn
import bitsandbytes as bnb
import lightning.pytorch as pl
from torch.utils.cpp_extension import load
from torch.nn import functional as F
from lightning.pytorch.strategies import DeepSpeedStrategy

if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# JIT 