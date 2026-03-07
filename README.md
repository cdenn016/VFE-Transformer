# Gauge-Theoretic Transformer - CURRENTLY IN FLUX

A research framework implementing gauge-covariant variational free energy (VFE) minimization for language modeling. This codebase accompanies the manuscript:

> **Attention, Transformers, and Backpropagation are Degenerate Limits of the Variational Free Energy Principle**
> Robert C. Dennis

## Core Thesis

                    h (hyper-prior)
                    │
                    │ KL(s||h) — regularization
                    ▼
    s_i = self.prior_mu[i]  ◄──── γ·KL(s_i||Ω_ij·s_j) ────► s_j
    (position MODEL,                    (model coupling)
     slow timescale)
                    │
                    │ p = w·π_token + (1-w)·s
                    ▼
              p_i (PRIOR)
                    │
                    │ α·KL(q||p)
                    ▼
         q_i (beliefs, fast)  ◄──── β_ij·KL(q_i||Ω_ij·q_j) ────► q_j
                                          (attention)



Language is a dynamic informational system: speakers encode and decode beliefs under uncertainty, and language models learn the statistical structure of this process. The mathematical framework natural to such systems---gauge-covariant variational free energy minimization over communicating agents on a statistical fiber bundle---explains why attention mechanisms work.

Standard transformer attention, the `1/sqrt(d_k)` scaling, layer normalization, and backpropagation all emerge as consequences of a single variational principle. The standard attention rule

```
beta_ij = softmax(Q_i K_j^T / sqrt(d_k))
```

is recovered as a **degenerate limit** of the gauge-theoretic attention

```
beta_ij = softmax(-D_KL(q_i || Omega_ij q_j) / tau)
```

through three successive simplifications: isotropic covariances, flat bundle, and learned projections.

## Theoretical Framework

### GL(K) Gauge Invariance (Theorem 1.1)

The KL divergence possesses maximal gauge symmetry: it is invariant under the full general linear group GL(K), not merely orthogonal subgroups.

```
D_KL(Omega_* P || Omega_* Q) = D_KL(P || Q)    for any invertible Omega in GL(K)
```

The (det Omega)^2 factors cancel in the log-determinant ratio. This means:
- Transport operators need only be **invertible**, not orthogonal
- No expensive re-orthogonalization is needed
- Learned projections W_Q, W_K in standard transformers **are themselves gauge transformations**, with `Omega = W_Q W_K^T` serving as a learned gauge transport

### Variational Free Energy

Each agent (token) carries a Gaussian belief `q_i = N(mu_i, Sigma_i)` in a local gauge frame `phi_i`. The full VFE is:

```
F = sum_i  D_KL(q_i || p_i)                              [belief-to-prior regularization]
  + sum_i  D_KL(s_i || r_i)                              [model-to-hyperprior regularization]
  + sum_ij beta_ij D_KL(q_i || Omega_ij q_j)             [belief alignment / attention]
  + sum_ij gamma_ij D_KL(s_i || Omega_ij s_j)            [model alignment / meta-cognition]
  - E_q[log p(o | {k_i}, {m_i})]                         [observation likelihood]
```

Attention weights emerge from information geometry:

```
beta_ij = softmax_j(-D_KL(q_i || Omega_ij q_j) / tau)
```

where `Omega_ij = exp(phi_i) exp(-phi_j)` is the gauge transport between agents. **No W_Q, W_K, W_V projections are used**---attention arises from the geometry of belief distributions.

### Three Limits Recovering Standard Attention

| Limit | What it discards | Result |
|---|---|---|
| **1. Isotropic covariances** | Non-isotropic Sigma_i -> sigma^2 I | KL reduces to squared Euclidean distance |
| **2. Flat bundle** | Position-dependent Omega_ij -> constant Omega | Global gauge, no curvature |
| **3. Learned projections** | sigma^{-2} Omega absorbed into W_Q W_K^T | Standard Q, K, V projections |

After all three limits: `beta_ij = softmax(Q_i K_j^T / sqrt(d_k))` --- standard transformer attention.

### Gradient Descent as Variational Inference

Gradient descent on the gauge-equivariant free energy recovers the standard transformer training update:

| FEP Framework | Neural Network |
|---|---|
| Free energy F[{q_i}] | Loss L(theta) |
| Belief q_i = N(mu_i, sigma^2 I) | Embedding h_i |
| Gauge transport Omega in GL(K) | W_Q W_K^T (learned) |
| -E_q[log p(o \| k)] | Cross-entropy loss |
| Vacuum (no observations) | Untrained network |
| Symmetry breaking | Training / learning |

### Multi-Head Attention as Block-Diagonal GL(K)

Multi-head attention restricts the full GL(d_k) to a block-diagonal subgroup:

```
G_multi-head = GL(d_head)^H  subset  GL(d_k)
```

Each head learns an independent GL(d_head) gauge transformation. For compact groups like SO(3), irreducible representations yield non-uniform head dimensions (1, 3, 5, 7, ...) with intrinsic geometric meaning.

Cross-head coupling extends this to sparse off-diagonal GL mixing between selected head pairs, enabling gauge transport across representation blocks.

### Multi-Timescale Dynamics

The free energy naturally separates into:
- **Fast (E-step)**: Belief inference `dq_i/dt = -eta_q dF_fast/dq_i` --- what transformers do in a forward pass
- **Slow (M-step)**: Model learning `ds_i/dt = -eta_s dF_slow/ds_i` --- what backpropagation updates

Standard transformers operate in the adiabatic limit: slow variables frozen during inference, updated between passes.

### Symmetry Breaking

Without observations, the free energy defines a gauge-symmetric vacuum---all agents converge to identical beliefs modulo gauge orbit (analogous to an untrained network). Observations break this symmetry, driving agents toward specialized representations determined by training data. Learning is thus interpreted as explicit symmetry breaking.

## Experimental Results

### BERT Validation (144 attention heads)

Quantitative comparison between gauge-aligned KL attention and standard dot-product attention on a pretrained `bert-base-uncased`:

| Metric | Value |
|---|---|
| Optimal temperature | tau = 19.0 (theory: tau = 2 sqrt(d) = 16, 19% deviation) |
| Mean Pearson correlation | r = 0.821 |
| Median Pearson correlation | r = 0.889 |
| Heads with r > 0.8 | 68.1% |
| Heads with r > 0.9 | 49.3% |

**Key-norm bias prediction confirmed**: average rho = -0.352 across all heads, significant in 92.4% of heads at p < 0.001. This gauge-theoretic prediction explains why layer normalization is a geometric necessity: it enforces constant key norms required for frame-independent inference.

### GL(30) Language Modeling (WikiText-103, vocab 50,257)

The **most general form** of the theory---no simplifying limits taken. Full non-isotropic covariances, non-trivial gauge transport, KL-divergence attention. **No MLPs, activation functions, learned W_Q/W_K/W_V, or positional encodings.** Only a linear output projection (from K=30 dimensions to 50k) is retained.

| Configuration | Layers | Gauge Mode | Train PPL | Test PPL | Parameters |
|---|---|---|---|---|---|
| Trivial gauge | 3 | Omega in GL(30) | 125.1 | 135.3 | ~50M |
| Learned gauge | 1 | Omega in GL(30) | 113.9 | 151.8 | ~50M |

For context: random-chance perplexity is ~50,000. These models capture substantial linguistic structure from gauge-theoretic principles alone.

**Emergent semantic structure**: Learned gauge frames develop interpretable categorical organization---punctuation, content words, and letters cluster separately in both belief space (mu) and gauge frame space (phi)---without any category supervision.

**Depth vs. gauge expressiveness**: The 3-layer trivial-gauge model generalizes better, while the 1-layer learned-gauge model fits training data better. Architectural depth and gauge expressiveness are partially substitutable.

## Installation

**Requirements**: Python 3.9+ with CUDA-capable GPU (recommended)

```bash
git clone https://github.com/cdenn016/VFE-Transformer.git
cd VFE-Transformer
pip install torch numpy scipy matplotlib seaborn plotly scikit-learn datasets tiktoken
```

### Optional Dependencies

- **Numba** --- JIT compilation for transport kernels (80x speedup over NumPy fallback)
- **NetworkX** --- Graph operations for RG flow analysis
- **Weights & Biases** (`wandb`) --- Experiment tracking

## Usage

### Training

```bash
# Standard VFE training with gauge-theoretic attention
python transformer/train.py

# Publication-quality training with all experimental features
python transformer/train_publication.py

# Resume from checkpoint after interruption
python transformer/resume_training.py
```

### Inference and Generation

```bash
# Text generation from trained model
python generate.py

# Model inference and analysis
python inference.py --checkpoint path/to/model.pt --prompt "The quick brown"

# Generate multiple samples
python inference.py --checkpoint path/to/model.pt --prompt "AI is" --num_samples 5

# Visualize attention patterns
python inference.py --checkpoint path/to/model.pt --prompt "The cat sat" --visualize
```

### Analysis

```bash
# RG flow analysis on trained model
python scripts/analyze_rg_flow.py --checkpoint path/to/model.pt --output rg_analysis/

# KN-5 baseline comparison (WikiText-103, matched BPE tokenization)
python scripts/kn5_baseline.py

# Publication figures
python scripts/generate_publication_figures.py
```

### Python API

```python
from transformer import GaugeTransformerLM, Trainer, TrainingConfig
from transformer.training.config import get_vfe_dynamic_config

# Create model
config = {
    'vocab_size': 257,        # Byte-level
    'embed_dim': 64,          # K = 64
    'n_layers': 3,
    'irrep_spec': [('fund', 4, 16)],  # 4 heads x GL(16)
    'hidden_dim': 256,
    'max_seq_len': 256,
    'kappa_beta': 1.0,
    'gauge_group': 'GLK',
    'gauge_mode': 'learned',  # or 'trivial' for Omega = I
    'ffn_mode': 'VFE_dynamic',
}
model = GaugeTransformerLM(config)

# Training config
train_config = get_vfe_dynamic_config(
    batch_size=16,
    max_steps=50000,
    use_wandb=False,
)

# Train
trainer = Trainer(model=model, config=train_config)
trainer.train(train_loader, val_loader)
```

## Project Structure

```
VFE-Transformer/
├── transformer/                   # Gauge-theoretic transformer (~55k lines total)
│   ├── core/                      # Core model components
│   │   ├── model.py               #   GaugeTransformerLM (full language model)
│   │   ├── attention.py           #   KL-divergence multi-head attention
│   │   ├── blocks.py              #   Transformer block and stack
│   │   ├── embeddings.py          #   Gauge token/positional embeddings
│   │   ├── variational_ffn.py     #   VFE feedforward (VFE_dynamic, pure_fep)
│   │   ├── ffn.py                 #   Standard gauge FFN
│   │   ├── prior_bank.py          #   Token-dependent prior bank for pure FEP
│   │   ├── gauge_utils.py         #   Shared matrix exp and KL utilities
│   │   └── sanitization.py        #   Numerical sanitization tracker
│   ├── training/                  # Training infrastructure
│   │   ├── config.py              #   TrainingConfig + preset configs
│   │   ├── optimizer.py           #   Per-parameter-group optimizer creation
│   │   └── metrics.py             #   Metrics tracking
│   ├── analysis/                  # Analysis and metrics
│   │   ├── rg_metrics.py          #   Renormalization group diagnostics
│   │   ├── rg_flow_analysis.py    #   RG flow analysis
│   │   ├── rg_flow_enhanced.py    #   Enhanced RG flow analysis
│   │   ├── publication_metrics.py #   Publication-quality metrics
│   │   ├── trajectory.py          #   Belief trajectory recording
│   │   └── semantics.py           #   Semantic structure analysis
│   ├── data/                      # Data loading
│   │   └── datasets.py            #   WikiText-2/103, byte/char/BPE dataloaders
│   ├── visualization/             # Plotting and visualization
│   │   ├── training_plots.py      #   Training curve plots
│   │   ├── trajectory_plots.py    #   Belief trajectory visualization
│   │   ├── attention_viz.py       #   Attention pattern visualization
│   │   ├── belief_space_viz.py    #   Belief space visualization
│   │   └── ablation_plots.py      #   Ablation study plots
│   ├── baselines/                 # Baseline models
│   │   └── standard_transformer.py#   Standard dot-product attention transformer
│   ├── experimental/              # Experimental code
│   │   ├── fep_transformer.py     #   FEP transformer variant
│   │   ├── pure_fep_transformer.py#   Pure FEP (backprop-free) transformer
│   │   └── train_fep.py           #   FEP training scripts
│   ├── utils/                     # Utilities
│   │   ├── checkpoint.py          #   Model save/load, tokenizer access
│   │   ├── evaluation.py          #   Evaluation utilities
│   │   └── testing.py             #   Test utilities
│   ├── train.py                   # Main training loop with full VFE loss
│   ├── train_publication.py       # Publication-quality training script
│   └── resume_training.py         # Resume training from checkpoint
├── math_utils/                    # Mathematical primitives
│   ├── generators.py              #   SO(3)/SO(N)/GL(K) Lie algebra generators
│   ├── transport.py               #   Parallel transport Omega_ij = exp(phi_i)exp(-phi_j)
│   ├── fisher_metric.py           #   Fisher-Rao natural gradients
│   ├── push_pull.py               #   Gaussian pushforward under transport
│   ├── sigma.py                   #   Covariance matrix utilities
│   ├── numba_kernels.py           #   JIT-compiled transport kernels
│   ├── cuda_kernels.py            #   CUDA kernel implementations
│   └── transport_cache.py         #   Transport operator caching
├── scripts/                       # Analysis and baseline scripts
│   ├── analyze_rg_flow.py         #   RG flow analysis on trained models
│   ├── kn5_baseline.py            #   KN-5 n-gram baseline comparison
│   └── generate_publication_figures.py
├── Attention/                     # Manuscript and data
│   ├── GL(K)_attention.tex        #   Main JMLR paper
│   ├── references.bib             #   Bibliography
│   ├── Data/                      #   Experimental data
│   └── figs/                      #   Figures
├── tests/                         # Test suite (123 passing tests)
│   └── transformer/
│       ├── test_attention.py      #   KL attention, transport, aggregation
│       ├── test_blocks.py         #   Transformer block and stack
│       ├── test_cross_head_coupling.py  # Cross-head gauge coupling
│       ├── test_embeddings.py     #   Gauge embeddings
│       ├── test_model.py          #   Full model integration
│       ├── test_training.py       #   Training config, optimizer, metrics
│       ├── test_integration.py    #   End-to-end integration
│       ├── test_imports.py        #   Import paths and sanitization
│       └── test_data.py           #   Data loading
├── generate.py                    # Text generation with attention visualization
├── inference.py                   # Model inference and qualitative analysis
└── transformer_test.py            # Quick model sanity check
```

## Architecture Details

### KL-Based Attention (No W_Q, W_K)

```
Attention weights:
  beta_ij = softmax_j(-D_KL(q_i || Omega_ij q_j) / kappa)

where D_KL between Gaussians:
  D_KL(q_i || Omega_ij q_j) = 0.5 * [
      tr((Omega_ij Sigma_j Omega_ij^T)^{-1} Sigma_i)
    + (mu_i - Omega_ij mu_j)^T (Omega_ij Sigma_j Omega_ij^T)^{-1} (mu_i - Omega_ij mu_j)
    - K
    + log(det(Omega_ij Sigma_j Omega_ij^T) / det(Sigma_i))
  ]

Message aggregation (GL(K) metric-corrected):
  m_i = sum_j beta_ij Omega_ij mu_j
  S_i = sum_j beta_ij Omega_ij Sigma_j Omega_ij^T   [for GL(K)]
```

### Gauge Groups

| Gauge Group | Generators | Transport | Use Case |
|---|---|---|---|
| **SO(3)** | 3 skew-symmetric | Rotations in 3D | Minimal geometric model |
| **SO(N)** | N(N-1)/2 skew-symmetric | Rotations in N-D | Intermediate complexity |
| **GL(K)** | K^2 general | Full invertible transport | Maximum expressiveness |
| **GL(d)^H** | H * d^2 block-diagonal | Per-head transport | Multi-head attention |

### Covariance Modes

| Mode | Storage | KL Cost | When to Use |
|---|---|---|---|
| **Full** | K x K per token | O(K^3) Cholesky | Small K, maximum expressiveness |
| **Diagonal** | K per token | O(K) | Large K, memory-constrained |
| **Shared** | Single K x K | O(K^3) amortized | Parameter-efficient |
| **Gauge-fixed** | Rotated from base | O(K^3) | Enforce gauge covariance |

### FFN Modes

| Mode | Mechanism | Backprop? |
|---|---|---|
| **VFE_dynamic** | Iterative VFE minimization with natural gradients | Yes (outer loop) |
| **pure_fep** | Prior evolution via p-flow (no weight updates) | No |
| **standard** | Conventional MLP (for baseline comparison) | Yes |

### Training Modes

Three preset configurations via `transformer.training.config`:

- **`get_standard_config()`** --- Standard transformer baseline (no gauge theory, single LR)
- **`get_vfe_dynamic_config()`** --- Full gauge-theoretic VFE training with per-parameter-group LRs
- **`get_pure_fep_config()`** --- Pure Free Energy Principle (backprop-free learning)

### Two-Timescale Learning

**Fast (belief inference)**: Within each forward pass, beliefs evolve via natural gradient descent:
```
for step in range(belief_steps):
    beta_ij = softmax(-KL(q_i || Omega_ij q_j) / kappa)
    grad_q = dF/dq
    mu_q <- mu_q - mu_lr * Sigma_q @ grad_q    # natural gradient
```

**Slow (prior learning)**: Token priors update across training steps:
```
prior_bank[target_v] <- (1 - prior_lr) * prior_bank[target_v] + prior_lr * avg_belief
```

### Renormalization Group Analysis

The VFE has self-similar structure: meta-agents satisfy the same definition as individual agents.

```
Scale zeta=0:   Tokens q_i = N(mu_i, Sigma_i) interact via beta_ij
                        | clustering (KL -> 0 within groups)
Scale zeta=1:   Meta-agents q_A = N(mu_A, Sigma_A) interact via beta'_AB
                        | further clustering
Scale zeta=2:   Super-meta-agents ...
```

Detected via spectral clustering on the attention matrix with metrics: modularity Q(beta), effective rank, intra/inter-cluster KL divergence. See `scripts/analyze_rg_flow.py` for analysis tooling.

## Numerical Stability

Key design decisions for stable training with large gauge groups:

- **sqrt(K) attention scaling**: `logits = -KL / (kappa * sqrt(K))` prevents softmax saturation
- **Log-space sigma clamping**: Clamp `log_sigma` before `exp()` to preserve gradients at boundaries (not post-exp clamping which kills gradients)
- **Logdet clamping**: `log(diag(L).clamp(min=eps))` not `log(diag(L) + eps)` prevents negative log-determinants
- **KL ceiling**: Fixed ceiling of 100.0 per element prevents divergence
- **Per-parameter gradient clipping**: Independent budget per parameter group (mu, sigma, phi, ffn)
- **GL(K) transport**: No re-orthogonalization needed (only invertibility required)
- **Float64 upcasting**: Matrix exponentials upcast to float64 for K >= 8 to prevent Pade overflow
- **Cholesky with jitter escalation**: Progressive jitter (1e-6 to 1e-1) with eigenvalue fallback
- **Sanitization tracking**: All numerical interventions (clamping, fallbacks, NaN replacement) are counted and reported via `transformer.core.sanitization.san`

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/transformer/test_attention.py -v
pytest tests/transformer/test_cross_head_coupling.py -v
pytest tests/transformer/test_model.py -v

# Run by marker
pytest tests/ -m "not slow"
```

123 tests covering: KL attention, transport operators, message aggregation, GL(K) metric correction, transformer blocks, gauge embeddings, cross-head coupling, model integration, training config, optimizer creation, and import paths.

## Documentation

- `Attention/GL(K)_attention.tex` --- Main JMLR manuscript with complete theory and proofs
- `transformer/PURE_FEP_TRANSFORMER_OVERVIEW.md` --- Pure FEP architecture overview
- `claude.md` --- Architecture overview and code standards

## Citation

If this work is useful, please cite:

```bibtex
@article{dennis2025attention,
  title={Attention, Transformers, and Backpropagation are Degenerate Limits of the Variational Free Energy Principle},
  author={Dennis, Robert C.},
  journal={Preprint},
  year={2025}
}
```

## License

This project is for research purposes.
