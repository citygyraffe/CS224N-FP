import torch
from torch import nn

class LoRALayer(torch.nn.Module):
    def __init__(self, dim_input, dim_output, rank=4, alpha=2.0):
        super().__init__()

        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        # A is k * r matrix
        self.A = torch.nn.Parameter(torch.randn(dim_input, rank) * std_dev)
        # B is r * d matrix
        self.B = torch.nn.Parameter(torch.zeros(rank, dim_output))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRALayer(torch.nn.Module):
    def __init__(self, linear_layer : nn.Linear, rank, alpha):
        super().__init__()
        self.linear_layer = linear_layer
        for param in self.linear_layer.parameters():
            param.requires_grad = False # Freeze the original linear layer
        self.lora = LoRALayer(linear_layer.in_features, linear_layer.out_features, rank, alpha)

    def forward(self, x):
        return self.linear_layer(x) + self.lora(x)