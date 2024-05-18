from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer

DEBUG_OUTPUT = True

class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # To complete this implementation:
                # 1. Update the first and second moments of the gradients.
                # 2. Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3. Update parameters (p.data).
                # 4. Apply weight decay after the main gradient-based updates.
                # Refer to the default project handout for more details.

                ### TODO

                if DEBUG_OUTPUT:
                    print("Iteration: ", state.get("t", 0))
                    print("hyperparameters: ", group)
                    print("state: ", state)
                    print("------------------------------")

                    # This is here to debug and take a look to see if optimizer is working correctly
                    if state.get("t", 0) == 2:
                        raise NotImplementedError("AdamW in debug mode please set DEBUG_OUTPUT to False to run the optimizer.")

                
                #if not "params" in state: state["params"] = group["params"]

                beta1, beta2 = group["betas"]
                epsilon = group["eps"]

                # 1. Update the first and second moments of the gradients
                first_moment_biased = beta1 * state.get("first_moment_biased", 0) + (1 - beta1) * grad
                second_moment_biased = beta2 * state.get("second_moment_biased", 0) + (1 - beta2) * grad ** 2

                # 2. Apply bias correction
                first_moment_corrected = first_moment_biased / (1 - (beta1 ** state.get("t", 0)))
                second_moment_corrected = second_moment_biased / (1 - (beta2 ** state.get("t", 0)))

                # 3. Update parameters (p.data)
                p.data = p.data - alpha * first_moment_corrected / (torch.sqrt(second_moment_corrected) + epsilon)

                # 4. Apply weight decay
                p.data = p.data - group["weight_decay"] * p.data * alpha

                if DEBUG_OUTPUT:
                    print("first_moment_biased: ", first_moment_biased)
                    print("second_moment_biased: ", second_moment_biased)
                    print("first_moment_corrected: ", first_moment_corrected)
                    print("second_moment_corrected: ", second_moment_corrected)

                # Update state dictionary
                state["first_moment_biased"] = first_moment_biased
                state["second_moment_biased"] = second_moment_biased
                state["t"] = state.get("t", 0) + 1

        return loss
