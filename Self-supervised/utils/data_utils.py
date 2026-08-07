# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Callable, Sequence
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset
import collections
import numpy as np
import torch
from monai.data import *
import pickle
from monai.transforms import *
from math import *
from torch.utils.data import DataLoader, ConcatDataset
from utils.data_trans import *
from utils.data_utils_abdomen import get_ds_abdomen
from utils.data_utils_headneck import get_ds_headneck
from utils.data_utils_chest import get_ds_chest
from utils.perf_diagnostics import InstrumentedDataset


def get_loader(args):
    abdomen_ds = get_ds_abdomen(args)
    headneck_ds = get_ds_headneck(args)
    chest_ds = get_ds_chest(args)

    abdomen_ls = []
    for _ in range(8):
        abdomen_ls.append(abdomen_ds)
    abdomen_ds = ConcatDataset(abdomen_ls)

    headneck_ls = []
    for _ in range(8):
        headneck_ls.append(headneck_ds)
    headneck_ds = ConcatDataset(headneck_ls)

    train_ds = ConcatDataset(
        [abdomen_ds, headneck_ds,
         chest_ds
         ])
    diagnostic_event = None
    if getattr(args, "diagnose_gpu_gaps", False) and not getattr(
        args, "data_check_only", False
    ):
        context_name = getattr(args, "multiprocessing_context", None)
        process_context = torch.multiprocessing.get_context(context_name)
        diagnostic_event = process_context.Event()
        diagnostic_event.set()
        train_ds = InstrumentedDataset(train_ds, diagnostic_event)

    train_sampler = Sampler(train_ds) if args.distributed else None
    workers = int(args.workers)
    loader_kwargs = {}
    if workers > 0:
        loader_kwargs["prefetch_factor"] = int(getattr(args, "prefetch_factor", 2))
        context = getattr(args, "multiprocessing_context", None)
        if context is not None:
            loader_kwargs["multiprocessing_context"] = context
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        sampler=train_sampler,
        pin_memory=bool(getattr(args, "pin_memory", True)),
        persistent_workers=bool(
            workers > 0 and getattr(args, "persistent_workers", True)
        ),
        **loader_kwargs,
    )

    train_loader.voco_diagnostic_event = diagnostic_event
    return train_loader
