"""Exponential moving average of nn.Module parameters.

Minimal, no extra deps. Shadows trainable parameters only — not buffers
(RunningNormalizer stats / LayerNorm running stats already track their own
running averages; double-smoothing them hurts rather than helps).

Usage:
    policy.attach_ema(decay=0.999)
    ...
    policy.train_weighted(...)           # calls ema.update(self) internally
    ...
    with policy.ema_swapped_in():
        # inside this block, policy's live params are the EMA weights
        eval_stats = evaluate_policy(policy, ...)
        save_checkpoint(path, policy)    # persists the EMA snapshot

After the block exits, the original training weights are restored.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn


class EMA:
    """Parameter-only EMA tracker attached to a module.

    Kept separate from the module so the state_dict of the underlying policy
    is unchanged — EMA state is written to a parallel `"ema"` key by the
    holder when checkpointing is desired, or simply swapped in via
    `swapped_in()` before saving so the raw state_dict already contains the
    EMA weights.
    """

    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 <= decay < 1.0:
            raise ValueError(
                f"EMA decay must be in [0, 1), got {decay!r}. "
                "Use 0.0 to disable EMA; typical values are 0.99-0.9999."
            )
        self.decay = decay
        # Clone current params as the initial EMA snapshot (shadow ≡ model at t=0).
        # Only `requires_grad` tensors get shadowed — frozen buffers (e.g. obs
        # normalizer running stats, LayerNorm running stats, action-bound
        # registered buffers) follow their own update semantics.
        self._shadow: dict[str, torch.Tensor] = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        # Scratch space used by `swapped_in()` to restore training weights
        # after an eval/save window.
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend current params into the shadow: ema ← d·ema + (1-d)·θ."""
        d = self.decay
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            shadow = self._shadow.get(n)
            if shadow is None:
                # Parameter added after EMA was attached — seed it.
                self._shadow[n] = p.detach().clone()
                continue
            # In-place: mul_ + add_ avoids a temporary tensor per step.
            shadow.mul_(d).add_(p.detach(), alpha=1.0 - d)

    @contextmanager
    def swapped_in(self, model: nn.Module) -> Iterator[None]:
        """Context manager: temporarily replace model's params with EMA values.

        The training weights are restored on exit even if the caller raises.
        Safe to nest around `save_checkpoint` and `evaluate_policy` calls so
        the checkpoint on disk and the reported eval cost reflect the same
        (smoothed) policy.
        """
        if self._backup:
            raise RuntimeError(
                "EMA.swapped_in() is not re-entrant — nested swap requested."
            )
        self._backup = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        try:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in self._shadow:
                        p.data.copy_(self._shadow[n])
            yield
        finally:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in self._backup:
                        p.data.copy_(self._backup[n])
            self._backup.clear()

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Flat param-name → tensor, for optional checkpoint inclusion."""
        return {n: p.detach().clone() for n, p in self._shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Overwrite shadow values with `state` (in-place)."""
        for n, v in state.items():
            if n in self._shadow:
                self._shadow[n].copy_(v)

    @torch.no_grad()
    def l2_drift(self, model: nn.Module) -> float:
        """||θ - θ_ema||₂ across all shadowed params. Diagnostic only."""
        total = 0.0
        for n, p in model.named_parameters():
            if n in self._shadow:
                total += float(((p.detach() - self._shadow[n]) ** 2).sum().item())
        return total ** 0.5

    @torch.no_grad()
    def sync_to(self, model: nn.Module) -> None:
        """Hard-copy shadow → model params (θ ← θ_ema), in place.

        Unlike `swapped_in()` (transient), this is a permanent overwrite:
        after the call, θ and the shadow are equal, and subsequent training
        resumes from the smoothed weights. Used to implement "hard-sync"
        EMA in iter-boundary distillation loops, where the EMA is promoted
        to the actual policy at the end of each round so the next round's
        teacher / prior sees the stabilised weights (not the noisy training
        trajectory).

        Note: this does NOT reset the shadow. After calling, shadow == θ,
        and the next `update()` will begin decaying the shadow toward the
        new post-step θ as usual.

        Callers that pair this with an Adam reset should do the reset
        AFTER this call (so the new optimizer state is consistent with
        the post-sync θ, not the pre-sync θ).
        """
        for n, p in model.named_parameters():
            if n in self._shadow:
                p.data.copy_(self._shadow[n])
