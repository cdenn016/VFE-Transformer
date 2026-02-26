# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 19:06:36 2025

@author: chris and christine
"""

# VFE Transformer with Renormalization Group Analysis

## Project Overview

This is a **Gauge-Theoretic Transformer** implementing **Variational Free Energy (VFE)** minimization for active inference. The key innovation is replacing learned attention projections (W_Q, W_K) with **information-geometric attention based on KL divergence**, enabling principled belief evolution and uncertainty quantification.

### Core Insight: VFE as Renormalization Group

The VFE theory has a natural **self-similar structure**: meta-agents satisfy the same definition as agents. This is the defining property of the **Renormalization Group (RG)**.

```
Scale ζ=0:  Tokens q_i = N(μ_i, Σ_i)     interact via β_{ij}
                    ↓ clustering (KL → 0 within groups)
Scale ζ=1:  Meta-agents q_A = N(μ_A, Σ_A)  interact via β'_{AB}
                    ↓ further clustering
Scale ζ=2:  Super-meta-agents...
```

## Architecture

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| Transformer Block | `transformer/transformer_block.py` | Main block with gauge attention + FFN |
| Attention | `transformer/attention.py` | KL-divergence based attention (no W_Q, W_K!) |
| FFN | `transformer/ffn.py` | Unified FFN (learned, VFE, hamiltonian modes) |
| Variational FFN | `transformer/variational_ffn.py` | VFE implementations including VFE_dynamic |
| RG Metrics | `transformer/rg_metrics.py` | **NEW**: RG analysis tools |
| Model | `transformer/model.py` | Full language model |

### FFN Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `learned` | Standard MLP + GELU | Baseline |
| `variational_gradient_engine` / `VFE` | Fixed-β VFE descent | Standard active inference |
| `VFE_dynamic` | **Dynamic-β VFE** (β recomputed each step) | RG analysis, meta-agent emergence |
| `hamiltonian` | Symplectic Hamiltonian dynamics | Energy-conserving evolution |

## Key Equations

### Attention Weights
```
β_ij = softmax_j(-KL(q_i || Ω_ij[q_j]) / κ)
```

where:
- `q_i = N(μ_i, Σ_i)`: Agent i's belief distribution
- `Ω_ij = exp(φ_i·G)·exp(-φ_j·G)`: Parallel transport operator
- `κ`: Temperature parameter

### Free Energy
```
F = α·KL(q||p)                                    # Prior anchoring
  + λ_β·Σ_{i,j} β_ij·KL(q_i || Ω_ij[q_j])       # Belief alignment
  + CE(W_out·μ, targets)                          # Discrete observations
```

### The Nonlinearity (replaces GELU)
```
∂β_ij/∂μ_i = β_ij · [∂KL_ij/∂μ_i - Σ_k β_ik · ∂KL_ik/∂μ_i] / κ
```

This creates **positive feedback**: similar beliefs → higher β → pulled closer → clusters form!

## RG Analysis

### Expected Behavior

If the RG theory is correct, VFE_dynamic should show:

| Observable | Expected Trend | Meaning |
|------------|----------------|---------|
| Modularity Q(β) | ↑ Increasing | Block structure emerges |
| Effective rank | ↓ Decreasing | Fewer effective DOF |
| KL within clusters | ↓ Decreasing | Meta-agents tighten |
| KL between clusters | → Stable | Groups remain distinct |

### Usage

```python
from transformer.ffn import GaugeFFN

# Create VFE_dynamic FFN
ffn = GaugeFFN(
    embed_dim=64,
    hidden_dim=256,
    generators=generators,  # SO(3) generators
    mode='VFE_dynamic',
    n_iterations=100,       # Many steps for RG equilibration
    kappa=1.0,              # Temperature
    alpha=0.01,             # Prior anchoring
    lambda_belief=1.0,      # Belief alignment
)

# Run with RG analysis
mu_out, sigma_out, beta_history, rg_flow = ffn.forward_with_rg_analysis(
    mu=mu,
    mu_prior=mu_prior,
    phi=phi,
    sigma=sigma,
)

# Check RG behavior
print(rg_flow.get_rg_trends())
print(rg_flow.is_rg_behavior())
```

### Demo Script

```bash
python experiments/rg_analysis_demo.py --n_iterations 100 --save_plots
```

## File Structure

```
VFE-Transformer-Renormalization/
├── transformer/
│   ├── attention.py          # KL-based attention
│   ├── ffn.py                 # Unified FFN dispatcher
│   ├── variational_ffn.py    # VFE implementations
│   ├── rg_metrics.py         # RG analysis tools (NEW)
│   ├── transformer_block.py  # Main transformer block
│   ├── model.py              # Full LM
│   └── hamiltonian_ffn.py    # Hamiltonian dynamics
├── experiments/
│   └── rg_analysis_demo.py   # RG demo/test script (NEW)
├── gradients/
│   └── gradient_engine.py    # Validated gradient computation
├── math_utils/
│   ├── generators.py         # SO(3) generators
│   └── transport.py          # Gauge transport
└── tests/
    └── test_transformer.py   # Unit tests
```

## Important Notes

1. **No W_Q, W_K matrices**: Attention emerges from geometry
2. **Diagonal covariance mode**: Use `diagonal_covariance=True` for O(K) instead of O(K²)
3. **GPU support**: All operations are fully vectorized PyTorch
4. **VFE_dynamic is expensive**: O(n_iterations × N² × K) but theoretically sound

## Testing

```bash
# Run transformer tests
pytest tests/test_transformer.py -v

# Run RG metrics tests
python transformer/rg_metrics.py

# Run full RG analysis
python experiments/rg_analysis_demo.py
```

## Contributing

When working on this codebase:

1. **Preserve gauge equivariance**: Covariance transport must be Σ_transported = Ω @ Σ @ Ω^T
2. **Use natural gradients**: Project Euclidean gradients via Fisher metric
3. **Test RG behavior**: New features should maintain or improve RG trends
4. **Document mathematical formulas**: Include LaTeX-style notation in docstrings





# Claude Code Guidelines for Gauge Transformer Project

## Domain Expertise

Apply these when working on this codebase:

- **Differential Geometry**: SPD manifolds, geodesics, affine-invariant metrics, Lie theory, fiber bundles
- **Variational Inference**: KL divergence, free energy, ELBO, information geometry
- **Gauge Theory**: Symmetries, equivariance, parallel transport, irreps

- **Matrix/Linear Algebra**: Eigendecomposition, Kronecker products, matrix exponentials

## Code Standards

- Write modular, testable functions with type hints
- Docstrings should include LaTeX math where relevant
- Variable names should match paper notation (e.g., `mu_q` for μ_q, `Sigma` for Σ)
- Check tensor shapes at each step when debugging
- Verify gradient flow with small-dim smoke tests

## Numerical Stability

- Use log-sum-exp for softmax operations
- Epsilon padding for matrix inverses
- Eigenvalue clamping to maintain positive definiteness

## Testing

- Property-based tests for mathematical invariants (symmetry, PSD)
- NumPy reference implementations for comparison
- Edge case coverage (single agent, identity matrices, zero inputs)



### What Works on GPU

| Module | CUDA Status | Notes |
|--------|-------------|-------|
| `model.py` | Full | Standard PyTorch nn.Module |
| `hamiltonian_ffn.py` | Full | Uses `torch.linalg.matrix_exp`, `torch.linalg.eigh` |
| `attention.py` | Full | Vectorized PyTorch path auto-selected on CUDA |
| `embeddings.py` | Full | Standard `nn.Embedding`, `register_buffer` |
| `train.py` | Full | Modern AMP API (`torch.amp.autocast/GradScaler`) |





### Known Dependency Issues (Anaconda/Windows)

**PyArrow DLL Error**: The `transformers` package has a broken dependency chain on Anaconda/Windows:
```
transformers → sklearn → pyarrow → DLL load failed
```

**Solution**: Use `tiktoken` instead of `transformers` for BPE tokenization:
```bash
# In Anaconda environment (not system Python!)
pip install tiktoken
```

The `data.py` module automatically prefers tiktoken when available. It's OpenAI's fast BPE tokenizer with no heavy dependencies.

**Multiple Python Environments**: Spyder (Anaconda) may use a different Python than `pip`. Always install with:
```python
import sys
!{sys.executable} -m pip install tiktoken
```

## Communication Style

**Be direct:**
- State errors and concerns plainly without excessive hedging
- "This is wrong because X" not "This might potentially be slightly off"


**minimize itemizations when working on manuscripts:**
- utilize academic prose 
- minimize the usage of bullet points, itemizations, lists, etc.

**Push back:**
- Challenge gaps in derivations, ask for justification
- If a claim needs proof, ask for it

**Skip praise preambles:**
- No "Great question!" openers—just answer
- No "Excellent point!"—just engage with the substance

**Flag simpler alternatives:**
- Call out over-engineering
- Ask what complexity buys if something seems unnecessarily elaborate

**Maintain position under pushback:**
- Don't fold immediately when disagreeing
- Ask "What am I missing?" rather than capitulating

**Honest uncertainty:**
- "I'm not sure this is right" beats confident speculation
- Acknowledge when something needs verification

**No bullshit:**
- If a correspondence is interpretive rather than mathematically exact, say so explicitly
- If something doesn't connect, don't force it—admit the gap
- Remove content that doesn't earn its place through rigorous derivation
- Never dress up hand-waving as theorem
- When asked "what does X have to do with anything?"—if the answer is "not much", say that



### Categorical Observation Precision (Transformer-Specific)

For transformers with softmax output p = softmax(W_out @ μ / τ):

```
Λ_o = (1/τ²) W^T (diag(p) - pp^T) W = (1/τ²) Cov_p(W)
```

This is the **Hessian of cross-entropy** with respect to μ:
- When p is peaked (confident): Λ_o has low rank, weak constraint
- When p is uniform (uncertain): Λ_o reflects full embedding structure
- Temperature τ scales precision (lower τ → higher precision)

### The Nonlinearity

Standard transformer: GELU(x) — ad hoc, nobody knows why it works

Ours: ∂β_{ij}/∂θ — emerges from differentiating softmax attention:

```
β_{ij} = softmax(-KL_{ij} / κ)

∂β_{ij}/∂μ_i = β_{ij} · [∂KL_{ij}/∂μ_i - Σ_k β_{ik} · ∂KL_{ik}/∂μ_i] / κ
∂β_{ij}/∂Σ_i = β_{ij} · [∂KL_{ij}/∂Σ_i - Σ_k β_{ik} · ∂KL_{ik}/∂Σ_i] / κ
∂β_{ij}/∂φ_i = β_{ij} · [∂KL_{ij}/∂φ_i - Σ_k β_{ik} · ∂KL_{ik}/∂φ_i] / κ
```



## VFE, Renormalization, and the Information Bottleneck

The gauge transformer framework unifies three deep theoretical perspectives: variational inference, renormalization group flow, and the information bottleneck principle.

### The Information Bottleneck (IB) Principle

Tishby's IB: Find representation Z of input X that predicts Y while compressing:

```
L_IB = I(Z; Y) - β · I(Z; X)
```

- **I(Z; Y)**: Preserve information relevant to target (prediction)
- **I(Z; X)**: Discard irrelevant input details (compression)
- **β**: Tradeoff parameter

### VFE IS the Information Bottleneck

The variational free energy:

```
F = α·KL(q||p) + CE(Wμ, y) + λ·Σ_ij β_ij·KL(q_i||Ω_ij·q_j)
    ─────────   ──────────   ─────────────────────────────
    compression  prediction        coherence
```

| IB Term | VFE Term | Meaning |
|---------|----------|---------|
| I(Z; X) | KL(q ‖ p) | Bits used beyond prior |
| I(Z; Y) | -CE | Prediction accuracy |
| β | α | Compression-accuracy tradeoff |

**The prior p is the reference channel** — beliefs at p carry zero information, deviations carry bits.

### Dynamic β: Adaptive Compression

The attention weights:

```
β_ij = softmax(-KL(q_i || Ω_ij·q_j) / κ)
```

This implements **input-dependent compression**:

- **Similar beliefs** (low KL) → high β → pool/average information
- **Distinct beliefs** (high KL) → low β → preserve separately

**The temperature κ IS the IB tradeoff**:
- High κ → soft attention → aggressive compression
- Low κ → sharp attention → selective preservation

### Renormalization = Hierarchical IB

Each VFE iteration (or layer) performs coarse-graining:

```
Raw tokens:    [t1] [t2] [t3] [t4] [t5] [t6]
                    ↓ VFE step (β clusters similar beliefs)
Meta-agents:   [   A   ] [   B   ] [   C   ]
                    ↓ VFE step
Coarser:       [     X     ] [     Y     ]
                    ↓
Output:        Minimal sufficient statistics for prediction
```

- **Tokens with KL≈0 merge** — within-group variation discarded
- **Between-group differences survive** — predictively relevant
- **RG fixed point** = optimal IB representation (can't compress further without losing prediction)

### Gauge Invariance: Geometric Compression

The transport Ω_ij enforces gauge invariance:

```
KL(q_i || Ω_ij·q_j) is gauge-invariant
```

- Multiple configurations differing by gauge → same representation
- Gauge-variant information automatically compressed out
- Only gauge-invariant (physical) information survives

This is a **symmetry-based prior** implementing compression geometrically.

### The Unified Picture

| Concept | IB View | VFE View | RG View |
|---------|---------|----------|---------|
| Compression | min I(Z;X) | KL(q‖p) → 0 | Coarse-graining |
| Prediction | max I(Z;Y) | min CE | Fixed point stability |
| Tradeoff | β parameter | κ temperature | Relevant vs irrelevant |
| Representation | Z | (μ, Σ) beliefs | Renormalized couplings |
| Hierarchy | Deep IB | VFE iterations | RG flow |

**Key insight**: Emergent block structure in β_ij reveals which tokens carry redundant information about the target and can be safely merged. The dynamics discovers the optimal compression automatically.







## References

- Active Inference: Friston et al.
- Gauge Theory in ML: Bronstein et al.
- Renormalization Group: Wilson, Kadanoff

- Information Geometry: Amari


