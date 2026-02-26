# Pure FEP Transformer: Complete Architecture Overview

This document explains how the Pure FEP (Free Energy Principle) Transformer works - a transformer architecture that learns **entirely through Variational Free Energy (VFE) minimization**, without backpropagation or external optimizers.

## Table of Contents

1. [Core Philosophy](#core-philosophy)
2. [The Complete VFE Formula](#the-complete-vfe-formula)
3. [What Beliefs and Priors Represent](#what-beliefs-and-priors-represent)
4. [Architecture Components](#architecture-components)
5. [Two-Timescale Learning Dynamics](#two-timescale-learning-dynamics)
6. [How Attention Emerges](#how-attention-emerges)
7. [Meta-Agents and Hierarchical Structure](#meta-agents-and-hierarchical-structure)
8. [Gauge-Equivariant Position Encoding](#gauge-equivariant-position-encoding)
9. [Prior Bank and Output Modes](#prior-bank-and-output-modes)
10. [Data Flow Summary](#data-flow-summary)

---

## Core Philosophy

Standard transformers use backpropagation to adjust weights. The Pure FEP Transformer instead follows a biological principle: **minimize surprise**.

The brain is theorized to continuously minimize its "free energy" - a bound on surprise. The Pure FEP Transformer implements this:

1. **Beliefs (q)** represent what the model INFERS given observations
2. **Priors (p)** represent what the model EXPECTS before seeing observations
3. **Learning** = priors flowing toward posteriors (beliefs)

No `optimizer.step()`. No `.backward()`. Just VFE gradient descent on beliefs and slow prior evolution.

---

## The Complete VFE Formula

The transformer minimizes this objective:

```
F = alpha * Sum_i KL(q_i||p_i)                       [Self-coupling: belief-to-prior]
  + lambda_belief * Sum_i Sum_j beta_ij * KL(q_i||Omega_ij * q_j)  [Belief alignment: attention]
  + lambda_obs * Sum_i E_{q_i}[-log p(y_i|mu_i)]     [Observation likelihood: CE]
  + lambda_prior * Sum_i Sum_j KL(p_i||Omega_ij * p_j)  [Prior alignment: structure]
  + Sum_i Sum_d decay^d * KL(p_i||h_i^d)             [Ouroboros Tower: hyperpriors]
```

Where:
- `i, j` index positions (tokens/agents)
- `q_i` = belief distribution at position i (posterior)
- `p_i` = prior distribution at position i
- `beta_ij` = belief attention weights (dynamic, from current beliefs)
- `Omega_ij` = gauge transport operator from position j to i
- `h_i^d` = hyperprior at depth d (static or slow)
- `y_i` = observation (target token)

### What Each Term Does

| Term | Purpose | Timescale |
|------|---------|-----------|
| **KL(q\|\|p)** | Beliefs stay close to priors | Fast (inference) |
| **beta * KL(q\|\|Omega*q)** | Beliefs align with neighbors | Fast (inference) |
| **CE(mu, y)** | Beliefs explain observations | Fast (inference) |
| **KL(p\|\|Omega*p)** | Priors of related tokens align | Slow (learning) |
| **KL(p\|\|h)** | Priors anchored to hyperpriors | Slowest (structure) |

---

## What Beliefs and Priors Represent

### The Key Insight

```
mu_p (prior mean)    -> encodes WHAT token (content/identity)
phi   (gauge frame)  -> encodes WHERE in sequence (position)
```

The token embedding IS the prior. Position is encoded separately in the gauge frame.

### In Language Modeling

| Component | Symbol | Meaning |
|-----------|--------|---------|
| **Prior** | p_i = prior_bank[x_i] | "What I expect given input token x_i" |
| **Belief** | q_i (after VFE) | "What I infer given context + observations" |
| **Observation** | y_i = target | "The actual next token" |
| **Likelihood** | p(y\|mu) | "How well mu predicts y" (via KL to token priors) |

### The Belief is a BALANCE

During VFE dynamics, the belief mu_q finds equilibrium between competing forces:

```
        mu_p (prior)
           \
            \  KL(q||p) pulls toward prior
             \
              * <- mu_q lands HERE (balanced point)
             / \
   beta*align /   \ CE gradient
          /     \
   neighbors   embed[y] (observation)
```

The posterior is NOT the observation - it's a compromise that:
- Respects the prior (doesn't forget token identity)
- Aligns with context (attention to neighbors)
- Explains observations (predicts target)

---

## Architecture Components

### 1. PriorBank: Unified Embedding & Output

Each token v has a prior belief encoded in the PriorBank:
```
pi_v = N(mu_v, Sigma_v) where mu_v = prior_bank.prior_mu[v]
```

**Encoding (replaces nn.Embedding):**
```python
q(z_i) <- pi_{input_token[i]}  # Belief starts at prior
```

**Decoding (replaces nn.Linear):**
```python
# output_mode='kl_to_prior': KL-based likelihood
logits = -KL(q || pi_v) / tau  # For each vocab token v
```

### 2. Embedding Modes

The model supports different embedding modes via `embedding_mode`:

| Mode | Description |
|------|-------------|
| `'prior_bank'` | Principled FEP - use PriorBank for both encode and decode |
| `'learned'` | Standard nn.Embedding (ad hoc but faster convergence) |
| `'hybrid'` | nn.Embedding input + PriorBank output |

### 3. Output Modes

The model supports different output modes via `output_mode`:

| Mode | Formula | Description |
|------|---------|-------------|
| `'kl_to_prior'` | `logits = -KL(q \|\| pi_v) / tau` | Principled FEP - KL divergence to token priors |
| `'linear'` | `logits = mu_q @ W_out.T` | Standard linear projection (ad hoc) |
| `'both'` | Average of above | Blend for comparison |

### 4. Gauge Position Encoding

Position lives in the gauge frame phi, NOT in the belief mean mu:

```python
position_5: mu = [semantic content], phi = [0.1, 0.2, 0.3]
position_9: mu = [semantic content], phi = [0.5, 0.6, 0.7]
```

Same mu (token identity), different phi (position).

Position affects HOW beliefs interact through transport:
```
Omega_ij = exp(phi_i - phi_j)  # Relative position determines transport
```

---

## Two-Timescale Learning Dynamics

### Fast Timescale: Q-Flow (Inference)

Within a single forward pass, beliefs evolve to minimize F:

```python
for step in range(belief_steps):
    # Compute attention from current beliefs
    beta_ij = softmax(-KL(q_i || Omega_ij * q_j) / kappa)

    # Compute gradients (self + alignment + observation)
    grad_q = dF/dq

    # Natural gradient descent
    mu_q <- mu_q - mu_lr * Sigma_q * grad_q
```

This is **perception** - inferring the current situation.

### Slow Timescale: P-Flow (Learning)

After inference, priors slowly evolve. In pure FEP mode, TWO types of priors are updated:

**1. Position-Dependent Layer Priors:**
```python
# Each layer maintains position-dependent priors (buffers)
layer.prior_mu <- (1 - prior_lr) * layer.prior_mu + prior_lr * mu_q
```

**2. Token Priors (PriorBank) - CRITICAL FOR LEARNING:**
```python
# For each target token v seen in batch:
# Move its prior toward the belief that should predict it
prior_bank.prior_mu[v] <- (1 - prior_lr) * prior_bank.prior_mu[v] + prior_lr * avg_belief_for_v
```

The token prior update is weighted by inverse prediction error: beliefs with lower error have stronger influence.

This is **learning** - updating expectations based on experience.

### Why Two Timescales?

| Flow | What Updates | Rate | Purpose |
|------|--------------|------|---------|
| Q-flow | Beliefs q | Fast (mu_lr ~ 0.1) | Fit current observation |
| P-flow | Layer priors + Token priors | Slow (prior_lr ~ 0.01) | Accumulate knowledge |

The prior accumulates "where beliefs typically end up" across many examples.

---

## How Attention Emerges

### Attention from KL Divergence

No learned W_Q, W_K, W_V matrices. Attention emerges from belief geometry:

```python
# Step 1: Compute divergence between transported beliefs
KL_ij = KL(q_i || Omega_ij * q_j)

# Step 2: Convert to attention weights
beta_ij = softmax(-KL_ij / kappa)  # Low KL = high attention

# Step 3: Aggregate information
mu_aggregate = Sum_j beta_ij * Omega_ij * mu_j
```

Tokens with aligned beliefs (low KL after transport) attend to each other.

### Two Types of Attention

| Type | Formula | Meaning |
|------|---------|---------|
| **beta_ij (belief attention)** | softmax(-KL(q_i\|\|Omega*q_j)/kappa) | "These align NOW" (dynamic) |
| **gamma_ij (prior attention)** | From prior similarity | "These SHOULD align" (structural) |

Beta is computed fresh each forward pass from current beliefs.
Gamma is learned structure about which tokens are related.

---

## Meta-Agents and Hierarchical Structure

### What Are Meta-Agents?

When tokens share aligned beliefs/priors, they form a **meta-agent**:

```
beta matrix:
        cat  dog  sat  mat
cat   [  1   .8   .1   .1  ]
dog   [ .8    1   .1   .1  ]     <- "cat-dog" meta-agent
sat   [ .1   .1    1   .7  ]
mat   [ .1   .1   .7    1  ]     <- "sat-mat" meta-agent
```

The block structure in beta reveals meta-agents.

### Meta-Agents at Different Levels

| Level | What Aligns | Timescale | Example |
|-------|-------------|-----------|---------|
| Beliefs | High beta_ij now | Fast | "cat" and "dog" in this sentence |
| Priors | High gamma_ij learned | Slow | "cat" and "dog" are both animals |
| Hyperpriors | Shared h | Static | Abstract categories |

### Gauge Orbits as Categories

Meta-agents are **gauge orbits** in prior space:

```
"animal" = { p : p ~ Omega*p_cat ~ Omega'*p_dog ~ Omega''*p_bird ... }
```

Tokens whose priors are related by gauge transport form an equivalence class.

---

## Gauge-Equivariant Position Encoding

### Transport Operators

The gauge transport Omega_ij relates beliefs at different positions:

```
Omega_ij = exp(phi_i * G) * exp(-phi_j * G)
```

Where G are the SO(3) or SO(N) generators.

### What Transport Does

```
mu_transported = Omega_ij * mu_j           # Rotate belief j into frame i
Sigma_transported = Omega_ij * Sigma_j * Omega_ij^T  # Rotate covariance accordingly
```

### Translation Invariance

If we shift all frames by delta:
```
phi_i -> phi_i + delta  for all i
Omega_ij = exp(phi_i + delta) * exp(-phi_j - delta) = exp(phi_i) * exp(-phi_j)  # Unchanged!
```

Relative positions determine attention, not absolute positions.

---

## Prior Bank and Output Modes

### PriorBank Structure

The PriorBank holds token priors and supports two modes:

**Standard Mode (gauge_fixed_priors=False):**
```python
prior_mu: (vocab_size, embed_dim)      # Per-token prior means
log_prior_sigma: (vocab_size, embed_dim)  # Per-token prior log-variances
```

**Gauge-Fixed Mode (gauge_fixed_priors=True):**
```python
base_prior_mu: (embed_dim,)            # Single shared base prior
phi_embed: (vocab_size, phi_dim)       # Per-token gauge frames
# Token prior = R_v @ base_prior_mu where R_v = exp(phi_v * G)
```

### Why Token Prior Updates Matter

In pure FEP mode, output logits depend on KL divergence to token priors:
```python
logits = -KL(belief || prior_bank[token]) / tau
```

If token priors don't update, the model can't learn which beliefs correspond to which tokens. The prediction stays at random chance (~50k PPL on 50k vocab).

**Solution:** After VFE inference, update each target token's prior toward the belief that should predict it:
```python
# Weighted by inverse prediction error
prior_bank.prior_mu[target_v] <- lerp(prior_mu[target_v], avg_belief, prior_lr)
```

---

## Data Flow Summary

```
Input: token_ids (B, N)
         |
         v
+------------------------------------------+
|  PriorBank / Embedding                   |
|  mu_p, sigma_p <- prior_bank[token_ids]  |
|  (Initialize beliefs from token PRIORS)  |
+------------------------------------------+
         |
         v
+------------------------------------------+
|  Position Encoding                       |
|  phi = position_encode(positions)        |
|  (Position in gauge frame, NOT in mu)    |
+------------------------------------------+
         |
         v
+------------------------------------------+
|  VFE Dynamics (Q-FLOW) - Per Layer       |
|  For step in range(belief_steps):        |
|    1. beta_ij = softmax(-KL(q||Omega*q)) |
|    2. grad = dF/dq (prior + align + CE)  |
|    3. mu_q <- mu_q - mu_lr * Sigma * grad|
|  Output: mu_q (balanced posterior)       |
+------------------------------------------+
         |
         v
+------------------------------------------+
|  Output Logits                           |
|  if output_mode == 'kl_to_prior':        |
|    logits = -KL(q || pi_v) / tau         |
|  elif output_mode == 'linear':           |
|    logits = mu_q @ W_out.T               |
+------------------------------------------+
         |
         v
+------------------------------------------+
|  Learning (P-FLOW)                       |
|  1. Update layer position priors:        |
|     layer.prior_mu <- lerp(prior, belief)|
|  2. Update token priors (CRITICAL!):     |
|     prior_bank[target] <- lerp(prior,    |
|                                avg_belief)|
+------------------------------------------+
         |
         v
Output: logits (B, N, vocab_size)
```

---

## The Complete Learning Picture

```
                    HYPERPRIORS (h)
                         |
                         | KL(p||h) - anchor
                         v
    +----------------------------------------------+
    |           PRIORS                             |
    |  - Layer priors: position-dependent buffers  |
    |  - Token priors: prior_bank.prior_mu         |
    |                                              |
    |  <- prior coupling: KL(p||Omega*p)           |
    |  <- p-flow: p -> q (slow learning)           |
    +----------------------------------------------+
                         |
                         | KL(q||p) - initialization
                         v
    +----------------------------------------------+
    |          BELIEFS (q)                         |
    |                                              |
    |  <- beta*KL(q||Omega*q) - attention align    |
    |  <- CE(logits,y) - observation gradient      |
    |  <- VFE descent (fast inference)             |
    +----------------------------------------------+
                         |
                         | logits = -KL(q||pi) / tau
                         v
                   OBSERVATIONS (y)
```

**Three timescales:**
- **Beliefs (q):** Fastest - within forward pass (mu_lr ~ 0.1)
- **Priors (p):** Slow - across training steps (prior_lr ~ 0.01)
- **Hyperpriors (h):** Slowest/static - fixed structure

---

## Configuration Quick Reference

Key parameters for Pure FEP mode (see `PureFEPConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pure_fep_mode` | True | Enable pure FEP (no backprop) |
| `belief_steps` | 20 | VFE steps per forward pass |
| `mu_lr` | 0.1 | Belief mean learning rate |
| `sigma_lr` | 0.025 | Belief variance learning rate |
| `prior_lr` | 0.01 | Prior update rate (layer + token priors) |
| `prior_update_interval` | 1 | Update priors every N batches |
| `embedding_mode` | 'prior_bank' | 'prior_bank', 'learned', or 'hybrid' |
| `output_mode` | 'kl_to_prior' | 'kl_to_prior', 'linear', or 'both' |
| `output_tau` | 1.0 | Temperature for KL-based output |
| `alpha` | 0.1 | Self-coupling weight: KL(q\|\|p) |
| `lambda_belief` | 1.0 | Belief alignment weight |
| `lambda_obs` | 1.0 | Observation likelihood weight |
| `kappa` | 0.1 | Attention temperature (lower = sharper) |
| `position_mode` | 'gauge_frame' | 'gauge_frame', 'sinusoidal_mu', or 'both' |
| `gauge_group` | 'SO3' | 'SO3' or 'SON' |

### Advanced Features (disabled by default)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prior_coupling_enabled` | False | Enable prior-prior coupling term |
| `lambda_prior` | 0.1 | Weight for prior coupling |
| `enable_ouroboros_tower` | False | Enable hyperpriors from all ancestors |
| `tower_max_depth` | 3 | Ancestor levels for hyperpriors |
| `tower_decay` | 0.3 | Weight decay per hyperprior level |
| `dynamic_layers_enabled` | False | Allow layer spawning/merging |
| `gauge_evolution_enabled` | False | Evolve gauge frames over time |

---

## Summary

The Pure FEP Transformer derives everything from VFE minimization:

| Traditional | Pure FEP |
|-------------|----------|
| nn.Embedding | PriorBank (token priors) |
| Output projection | KL to token priors |
| W_Q, W_K, W_V | KL-based attention |
| Position sinusoids | Gauge frame phi |
| Backprop | P-flow (priors -> posteriors) |
| Adam/SGD | Natural gradient descent |
| Attention heads | Emergent meta-agents |

**Critical for Learning:** Both layer priors AND token priors must be updated during P-flow. Token prior updates are what allow the model to learn which beliefs correspond to which output tokens.

The result is a transformer that learns like a brain is theorized to: by minimizing surprise through belief-prior dynamics, with meta-agents emerging naturally from attention structure.
