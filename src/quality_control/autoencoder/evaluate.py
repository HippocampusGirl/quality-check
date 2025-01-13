import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    step_count: int,
    step_pruning: int,
) -> float:
    model = model.eval()

    numerator: float = 0.0
    denominator: int = 0

    for images, _, _ in tqdm(val_dataloader, leave=False):
        (reconstructed_images,) = model(images, return_dict=False)
        numerator += mse_loss(reconstructed_images, images).cpu().item()
        denominator += images.size(0)

        if denominator > 20:
            break

    return numerator / denominator
