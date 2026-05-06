"""Policy package — shared toggles + helpers used by both policy classes."""

# ---------------------------------------------------------------------------
# Action-space output normalization (toggle)
# ---------------------------------------------------------------------------
#
# When True, the policy network's action head outputs in *normalized* action
# space (per-dim ∈ ~[-1, 1]) and we apply a per-dim affine `a = scale·n + bias`
# at the policy boundary (`act_np`, `action()`) to recover the physical action.
# Inversely, training labels (which are physical actions from MPPI) are
# normalized before the loss is computed against the network output. This
# equalizes per-dim gradient contribution across heterogeneous action ranges
# — the motivating case is Adroit Relocate, where arm slides span ~0.2 m and
# fingers span ~2.0 rad (10× ratio). Without this normalization, an unweighted
# MSE / mean-distance loss is dominated by finger residuals and arm dims
# receive proportionally tiny gradients, producing a policy whose arm
# commands stay close to current joint positions → tiny actuator forces →
# the visible "arm too weak to carry" failure mode.
#
# When False, behavior is **byte-identical** to pre-normalization
# checkpoints: no `_act_scale` / `_act_bias` buffers are registered, the
# network output flows straight through `act_np`'s clip, and old `.pt`
# files load with `load_state_dict(strict=True)` exactly as before. Flip
# off when you need to load a checkpoint trained before this change.
#
# IMPORTANT: the toggle must match the value used at training time when
# loading a checkpoint. A model trained with USE_ACT_NORM=True has its
# network learning a normalized output distribution; loading those weights
# under =False would skip the affine and produce wrong-magnitude actions.
# A model trained with =False has no `_act_scale` / `_act_bias` buffers in
# its state_dict; loading under =True would fail strict load with missing
# keys. Either case is loud rather than silently wrong, which is the point.
USE_ACT_NORM = True
