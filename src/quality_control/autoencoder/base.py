import torch


class Model(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        learning_rate_scheduler: torch.optim.lr_scheduler.LambdaLR,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.learning_rate_scheduler = learning_rate_scheduler
