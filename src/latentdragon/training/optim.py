"""`Adam16`: Adam over 16-bit parameters, with the arithmetic in fp32."""

import torch
from torch import Tensor

from latentdragon_kernels import resolve

SUPPORTED = (torch.float16, torch.bfloat16, torch.float32)
SEED_STRIDE = 100003  # separates the rounding draws of two parameters


class Adam16(torch.optim.Optimizer):
    """Adam whose state lives in the parameter's dtype and whose update runs in fp32.

    Adam squares the gradient, and fp16 flushes those squares to zero, so the second moment
    is stored as its root and the update never rounds through 16 bits. `stochastic`
    rounds writebacks up or down by distance; nearest rounding drops any update under half
    a step of the 16-bit grid, which freezes larger weights.
    `found_inf`, a device flag, skips a step without a host read.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        stochastic: bool = True,
        seed: int = 0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            stochastic=stochastic,
            fused=True,
        )
        super().__init__(params, defaults)
        self.seed = seed
        self.found_inf: Tensor | None = None
        self.kernel = None

    def _state_of(self, param: Tensor) -> dict:
        """The moments of `param`, made on its first step."""
        state = self.state[param]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param)
            state["exp_rms"] = torch.zeros_like(param)
        return state

    @torch.no_grad()
    def step(self, closure=None):
        """One Adam step over every parameter with a gradient."""
        for group in self.param_groups:
            for index, param in enumerate(group["params"]):
                if param.grad is None:
                    continue
                if param.dtype not in SUPPORTED:
                    raise ValueError(f"Adam16 cannot update {param.dtype} parameters")

                self.kernel = self.kernel or resolve("adam16", param.device)
                bad = self.found_inf
                if bad is None:
                    bad = torch.zeros((), device=param.device)

                # Steps count on the host, so a skipped step still advances them
                state = self._state_of(param)
                state["step"] += 1
                self.kernel(
                    param,
                    param.grad.to(param.dtype),
                    state["exp_avg"],
                    state["exp_rms"],
                    bad,
                    state["step"],
                    group["lr"],
                    group["betas"],
                    group["eps"],
                    group["weight_decay"],
                    group["stochastic"] and param.dtype != torch.float32,
                    self.seed + index * SEED_STRIDE,
                )
