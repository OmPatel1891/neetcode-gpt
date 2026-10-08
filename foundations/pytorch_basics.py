import torch
import torch.nn
from torchtyping import TensorType

# Round all answers to 4 decimal places: torch.round(tensor, decimals=4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        M, N = to_reshape.shape
        reshape_ten = torch.reshape(to_reshape, (M * N // 2, 2))
        return torch.round(reshape_ten, decimals=4)

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        avg_ten = torch.mean(to_avg, dim=0)
        return torch.round(avg_ten, decimals=4)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        cat_ten = torch.cat((cat_one, cat_two), dim=1)
        return torch.round(cat_ten, decimals=4)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        loss = torch.nn.functional.mse_loss(prediction, target)
        return torch.round(loss, decimals=4)