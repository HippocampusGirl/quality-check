from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

T = TypeVar("T", bound=torch.nn.Module)


@dataclass
class Model(Generic[T]):
    model: T
    optimizer: torch.optim.Optimizer
    learning_rate_scheduler: torch.optim.lr_scheduler.LambdaLR
