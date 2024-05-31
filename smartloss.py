from typing import Union, Callable
from loss_functions import kl_loss, sym_kl_loss, js_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from itertools import count
from bert import BertModel


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d


def inf_norm(x):
    return torch.norm(x, p=float('inf'), dim=-1, keepdim=True)


# def get_smart_loss(model, eval_fn, input_ids, attention_mask, loss, lambda_, alpha):
#     embed = model.embeddings(input_ids)
#     def eval(embed):
#         #### TODO
#         return logits
#     smart_loss_fn = SMARTLoss(eval_fn=eval,
#                               loss_fn=nn.CrossEntropyLoss(),
#                               loss_last_fn=sym_kl_loss,
#                               noise_var=alpha)
#     logits = eval(eval_fn)
#     smart_loss = loss + lambda_ * smart_loss_fn(embed, logits)
#     return total_loss


class SMARTLoss(nn.Module):
    def __init__(
        self,
        model,
        batch_size,
        eval_fn: Callable,
        loss_fn: Callable,
        loss_last_fn: Callable = None,
        norm_fn: Callable = inf_norm,
        num_steps: int = 1,
        step_size: float = 1e-3,
        epsilon: float = 1e-6,
        noise_var: float = 1e-5
    ) -> None:
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.eval_fn = eval_fn
        self.loss_fn = loss_fn
        self.loss_last_fn = default(loss_last_fn, loss_fn)
        self.norm_fn = norm_fn
        self.num_steps = num_steps
        self.step_size = step_size
        self.epsilon = epsilon
        self.noise_var = noise_var

    def forward(self, input_ids1, mask1, state: Tensor, input_ids2=None, mask2=None) -> Tensor:
        # embed = self.model.bert(input_ids1, mask1, perturb=True)
        # noise = torch.randn_like(embed['pooler_output'], requires_grad=True) * self.noise_var
        # print(f"noise:", noise)

        # Indefinite loop with counter
        for i in count():
            # Compute perturbed embed and states
            # embed_perturbed = embed + noise
            state_perturbed = self.eval_fn(input_ids1, mask1, perturb=True) if self.eval_fn == self.model.predict_sentiment else self.eval_fn(input_ids1, mask1, input_ids2, mask2, perturb=True)
            # Return final loss if last step (undetached state)
            # print(f"state_perturbed:", state_perturbed)
            if i == self.num_steps:
                return self.loss_last_fn(state_perturbed, state) / self.batch_size
            # Compute perturbation loss (detached state)
            # loss = self.loss_fn(state_perturbed, state.detach()) / self.batch_size
            # print(f"loss:", loss)
            # # Compute noise gradient ∂loss/∂noise
            # noise_gradient, = torch.autograd.grad(loss, noise, allow_unused=True)
            # print(f"noise_gradient:", noise_gradient)
            # # Move noise towards gradient to change state as much as possible
            # step = noise + self.step_size * noise_gradient
            # # Normalize new noise step into norm induced ball
            # step_norm = self.norm_fn(step)
            # noise = step / (step_norm + self.epsilon)
            # # Reset noise gradients for next step
            # noise = noise.detach().requires_grad_()


# class MultitaskBERTWithSMART(nn.Module):
#     def __init__(self, model, lambda_reg=0.02):
#         super(MultitaskBERTWithSMART, self).__init__()
#         self.model = model
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.lambda_reg = lambda_reg

#     def forward(self, input_ids1, mask1, labels, batch_size,
#                 optimizer, alpha=0.0001, input_ids2=None, mask2=None):
#         def eval