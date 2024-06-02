from typing import Callable
import torch.nn as nn
import torch.nn.functional as F
from itertools import count


def sym_kl_loss(input, target, reduction='sum', alpha=1.0):
    return alpha * F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target.detach(), dim=-1),
        reduction=reduction,
    ) + F.kl_div(
        F.log_softmax(target, dim=-1),
        F.softmax(input.detach(), dim=-1),
        reduction=reduction,
    )


class SMARTLoss(nn.Module):
    def __init__(
        self,
        model,
        batch_size,
        eval_fn: Callable,
        loss_last_fn: Callable = None,
        num_steps: int = 1,
        noise_var: float = 1e-5
    ) -> None:
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.eval_fn = eval_fn
        self.loss_last_fn = loss_last_fn
        self.num_steps = num_steps
        self.noise_var = noise_var

    def forward(self, input_ids1, mask1, state, input_ids2=None, mask2=None):
        for i in count():
            state_perturbed = self.eval_fn(input_ids1, mask1, perturb=True) if self.eval_fn == self.model.predict_sentiment else self.eval_fn(input_ids1, mask1, input_ids2, mask2, perturb=True)
            if i == self.num_steps:
                return self.loss_last_fn(state_perturbed, state) / self.batch_size
