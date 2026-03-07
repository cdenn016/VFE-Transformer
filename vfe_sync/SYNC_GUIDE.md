# VFE-Transformer Sync Guide

## Overview

VFE-Transformer was forked from Gauge-Transformer on ~Feb 26, 2026. Since then,
Gauge-Transformer received 28+ commits with critical bug fixes that VFE-Transformer
is missing. This guide documents every fix and provides patches to apply them.

## Patch Application Order

Apply patches in numerical order. Some patches depend on earlier ones.

```bash
cd /path/to/VFE-Transformer
for patch in /path/to/vfe_sync/patches/*.patch; do
    echo "Applying: $(basename $patch)"
    git apply --check "$patch" 2>&1 && git apply "$patch" || echo "MANUAL FIX NEEDED: $patch"
done
```

**Note:** Some patches may not apply cleanly because VFE-Transformer made its own
changes (removed positional encoding, removed single-irrep fallback, hardcoded
use_multi_irrep=True, added sanitization.py). Those will need manual resolution.

---

## Fix Categories (Priority Order)

### CRITICAL BUG FIXES (Apply First)

#### 1. Sign Error in Softmax Coupling Gradient (Patch 01)
**File:** `transformer/core/variational_ffn.py`
**Bug:** `grad_deviation = grad_kl_per_pair_full - avg_grad.unsqueeze(2)` has wrong sign.
**Fix:** `grad_deviation = avg_grad.unsqueeze(2) - grad_kl_per_pair_full`
**Impact:** Dampened instead of amplified the softmax nonlinearity. Affects all
VFE_dynamic attention-weight gradients.
**Locations:** 4 sites in variational_ffn.py (one per compute path: full, diagonal,
block-diagonal, multi-irrep).

#### 2. KL Temperature Scaling (Patch 02)
**File:** `transformer/core/attention.py`, `transformer/core/variational_ffn.py`
**Bug:** `dim_scale = 2.0 * math.sqrt(max(K, 1))` — the factor of 2 is wrong.
**Fix:** `dim_scale = math.sqrt(max(K, 1))`
**Theory:** The ½ in KL is a constant absorbed into κ, not part of dimensional
scaling. Standard attention uses √d_k, and KL logit-difference std ∝ √K.
**Locations:** attention.py (1 site), variational_ffn.py (5 sites).

#### 3. Critical Bugs from Deep Audit (Patch 03)
- **Stale phi return:** `model.py` returned pre-FFN phi instead of post-FFN phi.
  Fix: `'phi': phi_ffn if phi_ffn is not None else phi` (2 sites in model.py)
- **Init order:** `model.py` called `self.apply(_init_weights)` after creating
  out_proj, overwriting embedding's calibrated init_std. Fix: moved init before
  weight tying.
- **Cross-head attrs:** Added `self._cross_head_perm = None` etc. to `__init__`
  to prevent AttributeError.
- **KL quadratic optimization:** `attention.py` used two triangular solves where
  one suffices: `quad_term = torch.dot(y, y)` instead of solving L2^T z = y then
  dotting with delta_mu.
- **Position encoding squeeze:** `embeddings.py` used `.squeeze()` which fails
  when max_len=1. Fix: `.squeeze(-1)` to only remove last dim.

#### 4. Pad Masking in Attention (Patch 04)
**File:** `transformer/core/attention.py`
**Bug:** Beta clamping applied to masked (zero) positions, breaking causal mask zeros.
**Fix:**
```python
masked_positions = (logits == float('-inf'))
beta = torch.where(masked_positions, beta, beta.clamp(min=epsilon))
beta_sum = beta.sum(dim=-1, keepdim=True).clamp(min=epsilon)
beta = beta / beta_sum
```

#### 5. Division-by-Zero Guards (Patch 04)
**File:** `transformer/core/variational_ffn.py`
- Added `max(kappa * math.sqrt(max(K, 1)), eps)` at 6 sites
- Added `.clamp(min=eps)` before `torch.log()` in diagonal KL (3 sites)
- Added `.clamp(min=eps)` in logdet computations (3 sites)

#### 6. In-Place Operation Fixes (Patch 04)
**File:** `transformer/core/variational_ffn.py`
- `+=` on tensor slice breaks autograd. Fix: use `= (... + ...)` form.
**File:** `transformer/core/embeddings.py`
- `self.mu_embed.weight[id] = val` breaks autograd. Fix: use `.weight.data[id]`.

### NUMERICAL STABILITY FIXES

#### 7. Cholesky/Numerical Hardening (Patches 09, 10, 11, 13, 14)
Multiple commits hardened numerical paths:
- Progressive Cholesky regularization (replaces eigh fallback)
- Adaptive regularization for singular matrix crashes
- NameError fix for sigma_j_reg/I_d after _safe_spd_inv
- Phi norm clamping centralized in stable_matrix_exp_pair

#### 8. Diagonal Covariance Performance (Patches 12a, 12b, 12c)
Fixed diagonal covariance being ~50× slower than full covariance. Added
block-diagonal processing path.

### CODE QUALITY FIXES

#### 9. Import Cleanup (Patch 05)
- blocks.py: removed unused `import math` and `import torch.nn.functional as F`
- embeddings.py: replaced `np.log()` → `math.log()` (4 sites)
- prior_bank.py: replaced `np.sqrt()` → `math.sqrt()`
- Removes numpy dependency for scalar ops

#### 10. Dead Code Removal (Patch 06)
- variational_ffn.py: removed `_sanitize_euclidean_gradients()`,
  `_compute_cholesky_robust()`, and `MockMultiAgentSystem` (~110 lines)
- blocks.py: fixed misleading docstring ("For variational FFN" → "Required for VFE_dynamic mode")

#### 11. Test Fixes (Patch 07)
- test_model.py: fixed duplicate `'use_diagonal_covariance'` key and bad indentation

#### 12. Eval Config Fix (Patch 17)
- evaluation.py: added `'diagonal_covariance'` config key for checkpoint compatibility

### NEW FEATURES

#### 13. Gauge Preconditioner (Patch 08) — NEW FILE
**File:** `transformer/core/gauge_preconditioner.py`
Implements Cartan decomposition-based gradient preconditioning for GL(K) gauge frames.
- `build_cartan_projector()`: decomposes gl(K) = so(K) ⊕ sym(K)
- `build_slk_projector()`: projects φ to traceless sl(K)
- `apply_slk_projection()`: ensures det(Ω_ij) = 1
- `apply_cartan_preconditioning()`: dampens non-compact gradient directions

#### 14. RoPE Implementation (Patch 15)
Position-dependent SO(2)^{K/2} gauge rotations.

#### 15. Numerical Monitor (Patch 16)
math_utils/numerical_monitor.py enhancements.

---

## VFE-Transformer-Specific Considerations

VFE-Transformer made these intentional changes that may conflict:
1. **Removed positional encoding** — patches touching pos_encoding code may not apply
2. **Hardcoded use_multi_irrep=True** — patches adding multi_irrep conditionals may conflict
3. **Added sanitization.py** — no conflict (new file in VFE only)
4. **Removed single-irrep fallback** — patches modifying irrep branching may conflict
5. **Removed early stopping** — checkpoint patches may partially conflict

For each conflict, take the Gauge-Transformer fix but preserve VFE-Transformer's
intentional architectural decisions.
