"""
Sharpness-Aware Minimization (SAM) optimizer wrapper.

Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization" (ICLR 2021).

SAM seeks parameters that lie in neighborhoods with uniformly low loss, which
empirically correlates with better OOD generalization. It works by:
  1. Computing the gradient at the current point.
  2. Taking a small ascent step (rho) to find the worst-case perturbation.
  3. Computing the gradient at the perturbed point.
  4. Using that gradient to update the original parameters.

Usage:
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05)

    # In training loop:
    loss = criterion(model(inputs), labels)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    # Second forward-backward pass
    criterion(model(inputs), labels).backward()
    optimizer.second_step(zero_grad=True)
"""
import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization wrapper around any base optimizer.

    Args:
        params: Model parameters (iterable).
        base_optimizer: An *instantiated* optimizer (e.g. AdamW).
        rho: Neighbourhood size for the perturbation (default 0.05).
        adaptive: Use adaptive SAM (scale rho per-parameter by |p|). Default False.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive)
        # We need the param list before calling super().__init__
        params = list(params)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Ascent step: perturb parameters toward the worst-case loss."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group.get("rho", self.defaults["rho"]) / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Save the original parameters
                self.state[p]["old_p"] = p.data.clone()
                # Compute perturbation
                e_w = (torch.pow(p, 2) if group.get("adaptive", False) else 1.0) * p.grad * scale
                p.add_(e_w)  # Ascend to the worst-case point
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Descent step: restore original params and apply the base optimizer update."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # Restore original parameters
        self.base_optimizer.step()  # Apply the gradient computed at the perturbed point
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """Standard step (not used in SAM's two-step protocol, but needed for compatibility)."""
        raise NotImplementedError(
            "SAM requires two steps: call first_step() then second_step(). "
            "Use the SAM training loop pattern instead of optimizer.step()."
        )

    def _grad_norm(self):
        """Compute the L2 norm of all gradients (shared across param groups)."""
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group.get("adaptive", False) else 1.0) * p.grad)
                .norm(p=2)
                .to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
