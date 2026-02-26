"""
Transport Operator Extraction from Pretrained Transformers
==========================================================

Three methods for extracting the effective transport T_{ij} from token j
to token i through a pretrained transformer:

Method 0: Attention path defect (scalar proxy, trivially cheap)
Method 1: Attention-decomposed transport (one forward pass, approximate)
Method 2: Jacobian probing (exact, P*N forward passes)

The effective transport T_{ij} = dh_i^{(L)} / dh_j^{(0)} captures how
a perturbation at position j in the input propagates to position i at the
output, including all attention, FFN, layer norm, and residual effects.

CAUSAL NOTE: For causal (autoregressive) models like GPT-2, T[i,j] is
nonzero only when j <= i (earlier tokens influence later ones, not vice
versa). Curvature is measured via *path composition defect* on ordered
triples a < b < c rather than closed loops:

    D_{abc} = T[c,b] @ T[b,a] @ pinv(T[c,a])

For flat transport, D = I (path doesn't matter). Curvature means D != I.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class TransportResult:
    """Container for extracted transport operators."""
    # T[i, j] is a (d, d) matrix: effective transport from j to i
    # Full shape: (N, N, d, d) where N = sequence length, d = hidden dim
    # For causal models, T[i, j] is nonzero only when j <= i
    transport: torch.Tensor  # (N, N, d, d)
    method: str
    n_tokens: int
    d_model: int
    metadata: dict


# =========================================================================
# Method 0: Attention path defect (scalar proxy for causal models)
# =========================================================================

def attention_path_defect(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Scalar proxy for holonomy in causal models.

    For ordered triples a < b < c, compare:
        direct:   alpha[c, a]           (c attends directly to a)
        indirect: alpha[c, b] * alpha[b, a]  (c -> b -> a two-hop)

    Path defect = |indirect - direct| / (indirect + direct + eps)

    For flat attention flow, multi-hop composition should equal direct
    attention. Deviation indicates path-dependent information routing.

    Args:
        model: HuggingFace GPT2Model with output_attentions
        input_ids: (1, N) token IDs
        attention_mask: (1, N) attention mask

    Returns:
        dict with:
            'attentions': tuple of (1, H, N, N) per layer
            'defect_per_layer': (L, num_triples) defect values
            'triples': list of (a, b, c) ordered index triples
    """
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    attentions = outputs.attentions  # tuple of (1, H, N, N)
    if attentions is None:
        raise RuntimeError(
            "Model did not return attentions. Ensure model.config.output_attentions = True "
            "or pass output_attentions=True. Got None from model forward pass."
        )
    N = input_ids.shape[1]

    # Average over heads
    attn_matrices = [a.squeeze(0).mean(dim=0) for a in attentions]

    # Ordered triples a < b < c (all causal edges are valid)
    triples = _sample_ordered_triples(N, max_triples=500)

    defect_per_layer = []
    for A in attn_matrices:
        defects = []
        for a, b, c in triples:
            direct = A[c, a]
            indirect = A[c, b] * A[b, a]
            denom = direct + indirect + 1e-12
            defects.append(abs(indirect - direct) / denom)
        defect_per_layer.append(torch.stack(defects))

    return {
        'attentions': attentions,
        'defect_per_layer': torch.stack(defect_per_layer),  # (L, T)
        'triples': triples,
    }


# Keep old name as alias for backward compat in experiment.py
attention_flow_asymmetry = attention_path_defect


# =========================================================================
# Method 1: Attention-decomposed transport (approximate)
# =========================================================================

def attention_decomposed_transport(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    layers: Optional[List[int]] = None,
) -> TransportResult:
    """
    Compute effective transport via attention weights and value projections.

    Per-layer transport from j to i:
        T_{ij}^{(l)} = sum_h alpha_{ij}^{(l,h)} * W_O^{(h)} @ W_V^{(h)}

    Plus identity on diagonal (residual connection).

    The full transport is the (i,j) block of the product of per-layer
    block matrices: T^{eff} = prod_l T^{(l)}.

    Ignores FFN nonlinearity and layer norm — fast but approximate.

    Args:
        model: HuggingFace GPT2Model
        input_ids: (1, N) token IDs
        layers: which layers to include (default: all)

    Returns:
        TransportResult with transport shape (N, N, d, d)
    """
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    attentions = outputs.attentions  # tuple of (1, H, N, N)
    if attentions is None:
        raise RuntimeError(
            "Model did not return attentions. Ensure model.config.output_attentions = True."
        )
    N = input_ids.shape[1]

    # Extract weight matrices
    transformer_blocks = model.h if hasattr(model, 'h') else model.transformer.h
    n_layers = len(transformer_blocks)
    if layers is None:
        layers = list(range(n_layers))

    d_model = transformer_blocks[0].attn.c_proj.weight.shape[0]
    device = input_ids.device

    # Build composed transport as block matrix product
    # T_composed[i,j] is (d, d) — start with identity
    T_composed = torch.zeros(N, N, d_model, d_model, device=device)
    for i in range(N):
        T_composed[i, i] = torch.eye(d_model, device=device)

    for l in layers:
        block = transformer_blocks[l]
        attn = block.attn

        # GPT-2 uses Conv1D: weight is (d_in, d_out), so transpose
        # c_attn projects to [Q, K, V] concatenated
        W_qkv = attn.c_attn.weight  # (d_model, 3*d_model)
        d_head = d_model // attn.num_heads
        n_heads = attn.num_heads

        # Extract W_V: last d_model columns
        W_V = W_qkv[:, 2*d_model:3*d_model]  # (d_model, d_model)

        # W_O = c_proj
        W_O = attn.c_proj.weight  # (d_model, d_model)

        # alpha: (1, H, N, N) -> (H, N, N)
        alpha = attentions[l].squeeze(0)

        # Precompute per-head WOV matrices
        WOV = []  # list of (d_model, d_model)
        for h in range(n_heads):
            sl = slice(h * d_head, (h + 1) * d_head)
            W_V_h = W_V[:, sl]  # (d_model, d_head)
            W_O_h = W_O[:, sl]  # (d_model, d_head) for Conv1D
            WOV.append(W_O_h @ W_V_h.T)  # (d_model, d_model)

        # Build per-layer transport: T^{(l)}_{ij} for all i,j
        T_layer = torch.zeros(N, N, d_model, d_model, device=device)

        # Residual connection on diagonal
        for i in range(N):
            T_layer[i, i] = torch.eye(d_model, device=device)

        # Attention contribution (vectorized over heads)
        for h in range(n_heads):
            # alpha[h]: (N, N), WOV[h]: (d, d)
            # T_layer[i,j] += alpha[h,i,j] * WOV[h]
            # Vectorized: (N, N, 1, 1) * (d, d) -> (N, N, d, d)
            T_layer += alpha[h].unsqueeze(-1).unsqueeze(-1) * WOV[h]

        # Compose: T_new[i,j] = sum_k T_layer[i,k] @ T_old[k,j]
        T_new = torch.einsum('ikab,kjbc->ijac', T_layer, T_composed)
        T_composed = T_new

    return TransportResult(
        transport=T_composed,
        method='attention_decomposed',
        n_tokens=N,
        d_model=d_model,
        metadata={'layers': layers, 'n_layers': n_layers},
    )


def per_layer_holonomy(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_triples: int = 300,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Compute per-layer path defect using hidden-state probes.

    Instead of comparing full d×d transport matrices (which saturate at
    sqrt(2) in d²-dimensional space), compares what transport does to
    the actual hidden states:

        v_direct   = T_l[c,a] @ h_l[a]
        v_indirect = T_l[c,b] @ (T_l[b,a] @ h_l[a])
        kappa = ||v_ind_hat - v_dir_hat||₂

    This projects the comparison to d=768 dimensions (meaningful variation)
    rather than d²=589,824 (everything saturates).

    Memory: O(n_tri * d) per layer instead of O(N² * d²).
    No (N,N,d,d) transport tensor is ever constructed.

    Returns dict with:
        'kappa_mean':      float, mean defect across layers and triples
        'kappa_per_layer': list of float, mean defect per layer
        'kappa_all':       (n_layers, n_triples) array
        'triples':         list of (a,b,c)
        'n_layers':        int
        'n_tokens':        int
        'd_model':         int
    """
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )

    attentions = outputs.attentions
    hidden_states = outputs.hidden_states  # tuple of (1, N, d), len = n_layers + 1
    if attentions is None:
        raise RuntimeError(
            "Model did not return attentions. Ensure model.config.output_attentions = True."
        )

    N = input_ids.shape[1]
    transformer_blocks = model.h if hasattr(model, 'h') else model.transformer.h
    n_layers = len(transformer_blocks)
    d_model = transformer_blocks[0].attn.c_proj.weight.shape[0]
    device = input_ids.device

    # Sample triples once
    triples = _sample_ordered_triples(N, max_triples=max_triples, seed=seed)
    n_tri = len(triples)
    if n_tri == 0:
        return {
            'kappa_mean': float('nan'), 'kappa_median': float('nan'),
            'kappa_std': float('nan'), 'kappa_max': float('nan'),
            'kappa_per_layer': [], 'kappa_all': np.zeros((n_layers, 0)),
            'triples': [], 'n_layers': n_layers, 'n_tokens': N, 'd_model': d_model,
        }

    # Precompute index tensors for batched gathering
    idx_a = torch.tensor([t[0] for t in triples], device=device)
    idx_b = torch.tensor([t[1] for t in triples], device=device)
    idx_c = torch.tensor([t[2] for t in triples], device=device)

    kappa_all = np.zeros((n_layers, n_tri))

    with torch.no_grad():
        for l in range(n_layers):
            block = transformer_blocks[l]
            attn = block.attn

            W_qkv = attn.c_attn.weight
            d_head = d_model // attn.num_heads
            n_heads = attn.num_heads
            W_V = W_qkv[:, 2*d_model:3*d_model]
            W_O = attn.c_proj.weight

            alpha = attentions[l].squeeze(0)  # (H, N, N)
            h_l = hidden_states[l].squeeze(0)  # (N, d) — input to this layer

            # Probe vectors: h[a] for each triple
            h_a = h_l[idx_a]  # (n_tri, d)

            # Precompute WOV_h for all heads (12 × 768×768 = 28MB)
            WOV = []
            for h in range(n_heads):
                sl = slice(h * d_head, (h + 1) * d_head)
                WOV.append(W_O[:, sl] @ W_V[:, sl].T)  # (d, d)

            # --- v_direct = T[c,a] @ h[a] and v_step1 = T[b,a] @ h[a] ---
            # T[i,j] @ v = sum_h α[h,i,j] * WOV_h @ v   (for i≠j, causal)
            v_direct = torch.zeros(n_tri, d_model, device=device)
            v_step1 = torch.zeros(n_tri, d_model, device=device)

            for h in range(n_heads):
                Wh_a = (WOV[h] @ h_a.T).T  # (n_tri, d)
                v_direct += alpha[h, idx_c, idx_a].unsqueeze(1) * Wh_a
                v_step1  += alpha[h, idx_b, idx_a].unsqueeze(1) * Wh_a

            # --- v_indirect = T[c,b] @ v_step1 ---
            v_indirect = torch.zeros(n_tri, d_model, device=device)
            for h in range(n_heads):
                Wh_step = (WOV[h] @ v_step1.T).T  # (n_tri, d)
                v_indirect += alpha[h, idx_c, idx_b].unsqueeze(1) * Wh_step

            del WOV

            # Unit-norm directional defect on transported vectors (d-dimensional)
            norm_dir = v_direct.norm(dim=1, keepdim=True).clamp(min=1e-30)
            norm_ind = v_indirect.norm(dim=1, keepdim=True).clamp(min=1e-30)

            v_dir_hat = v_direct / norm_dir
            v_ind_hat = v_indirect / norm_ind

            kappas = (v_ind_hat - v_dir_hat).norm(dim=1)  # (n_tri,)
            kappa_all[l] = kappas.cpu().numpy()

            del v_direct, v_step1, v_indirect, h_a

    # Aggregate
    mean_per_triple = np.nanmean(kappa_all, axis=0)
    kappa_per_layer = [float(np.nanmean(kappa_all[l])) for l in range(n_layers)]

    return {
        'kappa_mean': float(np.nanmean(mean_per_triple)),
        'kappa_median': float(np.nanmedian(mean_per_triple)),
        'kappa_std': float(np.nanstd(mean_per_triple)),
        'kappa_max': float(np.nanmax(mean_per_triple)),
        'kappa_per_layer': kappa_per_layer,
        'kappa_all': kappa_all,
        'triples': triples,
        'n_layers': n_layers,
        'n_tokens': N,
        'd_model': d_model,
    }


# Keep old function for backward compat (returns list of TransportResult)
def per_layer_transports(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> List[TransportResult]:
    """Legacy: use per_layer_holonomy() instead for speed."""
    result = per_layer_holonomy(model, input_ids, attention_mask)
    # Can't reconstruct full transports from summary — raise helpful error
    raise NotImplementedError(
        "per_layer_transports() is deprecated. Use per_layer_holonomy() directly."
    )


# =========================================================================
# Method 2: Jacobian holonomy (exact, JVP-based)
# =========================================================================

def jacobian_holonomy(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_triples: int = 300,
    seed: int = 42,
    epsilon: float = 1e-3,
) -> Dict[str, object]:
    """
    Compute holonomy using exact Jacobian transport via finite-difference JVPs.

    Uses the actual model (LayerNorm, MLP, residual — everything).
    No attention approximation, no algebraic mismatch.

    For each triple (a, b, c) with a < b < c:
      v_direct   = J[c,a] @ h[a]     (perturb at a, read response at c)
      v_step1    = J[b,a] @ h[a]     (same perturbation, read at b)
      v_indirect = J[c,b] @ v_step1  (perturb at b with v_step1, read at c)

    For flat transport: v_direct ≈ v_indirect (influence of a on c factors
    through b). For curved: they differ — path matters.

    Cost: 1 + |unique_a| + |unique_(a,b)| forward passes per sentence
          (typically 50-200 for N≈20 tokens with 300 triples).

    Returns dict with kappa (directional distance) and cos_sim (cosine
    similarity) between direct and indirect transported vectors.
    """
    device = input_ids.device
    N = input_ids.shape[1]

    # Get embedding layer
    if hasattr(model, 'wte'):
        wte, wpe = model.wte, model.wpe
    elif hasattr(model, 'transformer'):
        wte, wpe = model.transformer.wte, model.transformer.wpe
    else:
        raise ValueError("Cannot find embedding layers (wte/wpe)")

    # Clean forward pass
    with torch.no_grad():
        clean_embeds = wte(input_ids)  # (1, N, d)
        pos_ids = torch.arange(N, device=device).unsqueeze(0)
        clean_embeds = clean_embeds + wpe(pos_ids)
        clean_output = _forward_from_embeds(model, clean_embeds, attention_mask)

    d_model = clean_output.shape[-1]
    h0 = clean_embeds.squeeze(0)  # (N, d) input embeddings

    triples = _sample_ordered_triples(N, max_triples=max_triples, seed=seed)
    n_tri = len(triples)
    if n_tri == 0:
        return {
            'kappa_mean': float('nan'), 'kappa_median': float('nan'),
            'kappa_std': float('nan'), 'kappa_max': float('nan'),
            'cos_sim_mean': float('nan'), 'cos_sim_std': float('nan'),
            'kappa_all': np.zeros(0), 'cos_sim_all': np.zeros(0),
            'triples': [], 'n_tokens': N, 'd_model': d_model,
            'n_forward_passes': 1,
        }

    unique_a = sorted(set(t[0] for t in triples))
    unique_ab = sorted(set((t[0], t[1]) for t in triples))

    # Step 1: For each unique source a, JVP with probe h0[a]
    # Gives T[i,a] @ h0[a] for all positions i
    jvp_a = {}
    with torch.no_grad():
        for a in unique_a:
            probe = h0[a]
            pn = probe.norm()
            if pn < 1e-30:
                jvp_a[a] = torch.zeros(N, d_model, device=device)
                continue
            perturbed = clean_embeds.clone()
            perturbed[0, a] += epsilon * probe / pn
            out = _forward_from_embeds(model, perturbed, attention_mask)
            jvp_a[a] = ((out[0] - clean_output[0]) / epsilon) * pn

    # Step 2: For each unique (a,b), JVP at b with v_step1 = T[b,a]@h[a]
    # Gives T[c,b] @ v_step1 for all positions c
    jvp_ab = {}
    with torch.no_grad():
        for a, b in unique_ab:
            v_step1 = jvp_a[a][b]  # (d,) = T[b,a] @ h[a]
            sn = v_step1.norm()
            if sn < 1e-30:
                jvp_ab[(a, b)] = torch.zeros(N, d_model, device=device)
                continue
            perturbed = clean_embeds.clone()
            perturbed[0, b] += epsilon * v_step1 / sn
            out = _forward_from_embeds(model, perturbed, attention_mask)
            jvp_ab[(a, b)] = ((out[0] - clean_output[0]) / epsilon) * sn

    # Step 3: Per-triple cosine similarity and kappa
    kappas = np.zeros(n_tri)
    cos_sims = np.zeros(n_tri)

    for t, (a, b, c) in enumerate(triples):
        v_direct = jvp_a[a][c]        # T[c,a] @ h[a]
        v_indirect = jvp_ab[(a, b)][c]  # T[c,b] @ T[b,a] @ h[a]

        nd = v_direct.norm().item()
        ni = v_indirect.norm().item()

        if nd < 1e-30 or ni < 1e-30:
            kappas[t] = float('nan')
            cos_sims[t] = float('nan')
            continue

        cs = (v_direct @ v_indirect).item() / (nd * ni)
        cs = max(-1.0, min(1.0, cs))  # numerical clamp
        cos_sims[t] = cs
        kappas[t] = np.sqrt(max(0.0, 2.0 * (1.0 - cs)))

    finite = np.isfinite(kappas)
    ck = kappas[finite]
    cc = cos_sims[finite]

    n_fwd = 1 + len(unique_a) + len(unique_ab)

    return {
        'kappa_mean': float(np.mean(ck)) if len(ck) > 0 else float('nan'),
        'kappa_median': float(np.median(ck)) if len(ck) > 0 else float('nan'),
        'kappa_std': float(np.std(ck)) if len(ck) > 0 else float('nan'),
        'kappa_max': float(np.max(ck)) if len(ck) > 0 else float('nan'),
        'cos_sim_mean': float(np.mean(cc)) if len(cc) > 0 else float('nan'),
        'cos_sim_std': float(np.std(cc)) if len(cc) > 0 else float('nan'),
        'kappa_all': kappas,
        'cos_sim_all': cos_sims,
        'triples': triples,
        'n_tokens': N,
        'd_model': d_model,
        'n_forward_passes': n_fwd,
    }


# =========================================================================
# Method 2 (legacy): Full Jacobian reconstruction
# =========================================================================

def jacobian_transport(
    model,
    input_ids: torch.Tensor,
    embedding_layer: nn.Module,
    attention_mask: Optional[torch.Tensor] = None,
    n_probes: int = 50,
    epsilon: float = 1e-3,
    seed: int = 42,
) -> TransportResult:
    """
    Compute effective transport via perturbation probing.

    For each position j and probe direction e_k:
        1. Perturb input embedding at j by epsilon * e_k
        2. Forward pass
        3. Measure (output_perturbed - output_clean) / epsilon at all positions i
        4. This gives one column of T_{ij} per probe

    Reconstruct transport from P probes via least-squares.

    Args:
        model: HuggingFace GPT2Model
        input_ids: (1, N) token IDs
        embedding_layer: the embedding module (model.wte or model.transformer.wte)
        n_probes: number of random probe directions P
        epsilon: perturbation magnitude
        seed: random seed for probe directions

    Returns:
        TransportResult with transport shape (N, N, d, d)
    """
    device = input_ids.device
    N = input_ids.shape[1]

    # Get clean output
    with torch.no_grad():
        clean_embeds = embedding_layer(input_ids)  # (1, N, d)
        # Add position embeddings if GPT-2
        if hasattr(model, 'wpe') or (hasattr(model, 'transformer') and hasattr(model.transformer, 'wpe')):
            wpe = model.wpe if hasattr(model, 'wpe') else model.transformer.wpe
            pos_ids = torch.arange(N, device=device).unsqueeze(0)
            clean_embeds = clean_embeds + wpe(pos_ids)

        clean_output = _forward_from_embeds(model, clean_embeds, attention_mask)
        # clean_output: (1, N, d)

    d_model = clean_output.shape[-1]

    # Generate random probe directions
    rng = torch.Generator(device='cpu').manual_seed(seed)
    probes = torch.randn(n_probes, d_model, generator=rng, device=device)
    probes = probes / probes.norm(dim=-1, keepdim=True)  # unit vectors

    transport = torch.zeros(N, N, d_model, d_model, device=device)

    for j in range(N):
        # Collect responses for all probes at position j
        responses = []  # will be (P, N, d)

        for p in range(n_probes):
            perturbed_embeds = clean_embeds.clone()
            perturbed_embeds[0, j, :] += epsilon * probes[p]

            with torch.no_grad():
                perturbed_output = _forward_from_embeds(
                    model, perturbed_embeds, attention_mask
                )

            response = (perturbed_output[0] - clean_output[0]) / epsilon  # (N, d)
            responses.append(response)

        # responses: (P, N, d)
        responses = torch.stack(responses, dim=0)

        probes_pinv = torch.linalg.pinv(probes)  # (d, P)

        for i in range(N):
            transport[i, j] = (probes_pinv @ responses[:, i, :]).T

    return TransportResult(
        transport=transport,
        method='jacobian_probing',
        n_tokens=N,
        d_model=d_model,
        metadata={'n_probes': n_probes, 'epsilon': epsilon},
    )


# =========================================================================
# Helpers
# =========================================================================

def _forward_from_embeds(
    model,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run model forward pass from embeddings, return hidden states."""
    # Handle both raw GPT2Model and wrapped models
    if hasattr(model, 'transformer'):
        # GPT2LMHeadModel — use the inner transformer
        core = model.transformer
    else:
        core = model

    outputs = core(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
    )
    return outputs.last_hidden_state


def _sample_ordered_triples(
    N: int,
    max_triples: int = 500,
    seed: int = 42,
    positions: Optional[List[int]] = None,
) -> List[Tuple[int, int, int]]:
    """
    Sample ordered token index triples (a < b < c) for causal holonomy.

    For causal models, all three causal edges T[c,b], T[b,a], T[c,a]
    are nonzero when a < b < c, making the path defect well-defined.

    If positions is provided, only sample triples from those token positions
    (useful for phrase-localized measurement).
    """
    import itertools

    if positions is not None:
        pos = sorted(set(positions))
        all_triples = list(itertools.combinations(pos, 3))
    else:
        all_triples = list(itertools.combinations(range(N), 3))

    if len(all_triples) <= max_triples:
        return all_triples

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(all_triples), size=max_triples, replace=False)
    return [all_triples[i] for i in sorted(indices)]


# Keep old name for any callers
_sample_triangles = _sample_ordered_triples


# =========================================================================
# Layer-wise Jacobian holonomy (each layer = one VFE step)
# =========================================================================

def layerwise_jacobian_holonomy(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_triples: int = 300,
    seed: int = 42,
    epsilon: float = 1e-3,
    positions: Optional[List[int]] = None,
) -> Dict[str, object]:
    """
    Compute per-layer Jacobian holonomy — each layer as one VFE step.

    For each layer l, perturbs the hidden state ENTERING that layer and
    measures the response EXITING that layer. This gives the exact
    single-layer transport T_l[i,j] = dh_i^{(l+1)} / dh_j^{(l)},
    including attention, FFN, layer norm, and residual.

    For each triple (a,b,c) with a < b < c, at each layer:
      v_direct   = T_l[c,a] @ h_l[a]
      v_step1    = T_l[b,a] @ h_l[a]
      v_indirect = T_l[c,b] @ v_step1
      kappa_l    = ||hat(v_ind) - hat(v_dir)||_2

    If positions is provided, only sample triples from those token positions
    (phrase-localized measurement). The full forward pass still uses all
    tokens for context.

    Returns dict with:
        'kappa_per_layer':  list of float, mean kappa at each layer
        'kappa_all':        (n_layers, n_triples) array
        'kappa_mean':       float, grand mean across layers and triples
        'kappa_median':     float
        'kappa_std':        float
        'kappa_max':        float
        'cos_sim_per_layer': list of float, mean cosine similarity per layer
        'triples':          list of (a,b,c)
        'n_layers':         int
        'n_tokens':         int
        'd_model':          int
        'n_forward_passes': int
    """
    device = input_ids.device
    N = input_ids.shape[1]

    if hasattr(model, 'wte'):
        wte, wpe = model.wte, model.wpe
    elif hasattr(model, 'transformer'):
        wte, wpe = model.transformer.wte, model.transformer.wpe
    else:
        raise ValueError("Cannot find embedding layers")

    transformer_blocks = model.h if hasattr(model, 'h') else model.transformer.h
    n_layers = len(transformer_blocks)

    # Full forward pass to get all hidden states
    with torch.no_grad():
        embeds = wte(input_ids) + wpe(torch.arange(N, device=device).unsqueeze(0))
        outputs = model(
            inputs_embeds=embeds if not hasattr(model, 'transformer') else None,
            input_ids=input_ids if hasattr(model, 'transformer') else None,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    # hidden_states[l] = input to layer l (shape: (1, N, d))
    hidden_states = outputs.hidden_states  # len = n_layers + 1

    d_model = hidden_states[0].shape[-1]

    triples = _sample_ordered_triples(N, max_triples=max_triples, seed=seed,
                                       positions=positions)
    n_tri = len(triples)
    if n_tri == 0:
        return {
            'kappa_mean': float('nan'), 'kappa_median': float('nan'),
            'kappa_std': float('nan'), 'kappa_max': float('nan'),
            'kappa_per_layer': [], 'kappa_all': np.zeros((n_layers, 0)),
            'cos_sim_per_layer': [], 'cos_sim_all': np.zeros((n_layers, 0)),
            'triples': [], 'n_layers': n_layers, 'n_tokens': N,
            'd_model': d_model, 'n_forward_passes': 0,
        }

    kappa_all = np.zeros((n_layers, n_tri))
    cos_sim_all = np.zeros((n_layers, n_tri))
    total_fwd = 0

    for l in range(n_layers):
        h_in = hidden_states[l].detach().clone()  # (1, N, d)

        # Helper: run just layer l on a given input
        def run_layer(h_input):
            with torch.no_grad():
                block = transformer_blocks[l]
                # GPT-2 block expects (batch, seq, hidden) and returns tuple
                # Need to handle layer norm -> attn -> residual -> ln -> ffn -> residual
                out = block(h_input)
                return out[0]  # (1, N, d)

        clean_out = run_layer(h_in)  # (1, N, d)

        # For each triple, we need JVPs:
        #   T_l[c,a] @ h[a]: perturb at a, read at c
        #   T_l[b,a] @ h[a]: perturb at a, read at b
        #   T_l[c,b] @ v:    perturb at b with v, read at c

        # Collect unique source positions and (source, mid) pairs
        unique_a = sorted(set(t[0] for t in triples))
        unique_ab = sorted(set((t[0], t[1]) for t in triples))

        # Step 1: JVP at each unique source a with probe h_in[0,a]
        jvp_a = {}
        for a in unique_a:
            probe = h_in[0, a]  # (d,)
            pn = probe.norm()
            if pn < 1e-30:
                jvp_a[a] = torch.zeros(N, d_model, device=device)
                continue
            perturbed = h_in.clone()
            perturbed[0, a] += epsilon * probe / pn
            out = run_layer(perturbed)
            jvp_a[a] = ((out[0] - clean_out[0]) / epsilon) * pn  # (N, d)
            total_fwd += 1

        # Step 2: JVP at b with v_step1 = T_l[b,a] @ h[a]
        jvp_ab = {}
        for a, b in unique_ab:
            v_step1 = jvp_a[a][b]  # (d,)
            sn = v_step1.norm()
            if sn < 1e-30:
                jvp_ab[(a, b)] = torch.zeros(N, d_model, device=device)
                continue
            perturbed = h_in.clone()
            perturbed[0, b] += epsilon * v_step1 / sn
            out = run_layer(perturbed)
            jvp_ab[(a, b)] = ((out[0] - clean_out[0]) / epsilon) * sn  # (N, d)
            total_fwd += 1

        # Step 3: Per-triple kappa
        for t, (a, b, c) in enumerate(triples):
            v_direct = jvp_a[a][c]        # T_l[c,a] @ h[a]
            v_indirect = jvp_ab[(a, b)][c]  # T_l[c,b] @ T_l[b,a] @ h[a]

            nd = v_direct.norm().item()
            ni = v_indirect.norm().item()
            if nd < 1e-30 or ni < 1e-30:
                kappa_all[l, t] = float('nan')
                cos_sim_all[l, t] = float('nan')
                continue

            cs = (v_direct @ v_indirect).item() / (nd * ni)
            cs = max(-1.0, min(1.0, cs))
            cos_sim_all[l, t] = cs
            kappa_all[l, t] = np.sqrt(max(0.0, 2.0 * (1.0 - cs)))

    # Aggregates
    kappa_per_layer = [float(np.nanmean(kappa_all[l])) for l in range(n_layers)]
    cos_sim_per_layer = [float(np.nanmean(cos_sim_all[l])) for l in range(n_layers)]
    mean_per_triple = np.nanmean(kappa_all, axis=0)
    finite = np.isfinite(mean_per_triple)
    clean = mean_per_triple[finite]

    return {
        'kappa_mean': float(np.mean(clean)) if len(clean) > 0 else float('nan'),
        'kappa_median': float(np.median(clean)) if len(clean) > 0 else float('nan'),
        'kappa_std': float(np.std(clean)) if len(clean) > 0 else float('nan'),
        'kappa_max': float(np.max(clean)) if len(clean) > 0 else float('nan'),
        'kappa_per_layer': kappa_per_layer,
        'kappa_all': kappa_all,
        'cos_sim_per_layer': cos_sim_per_layer,
        'cos_sim_all': cos_sim_all,
        'triples': triples,
        'n_layers': n_layers,
        'n_tokens': N,
        'd_model': d_model,
        'n_forward_passes': total_fwd,
    }


# =========================================================================
# Discrete Riemann curvature (composition non-additivity)
# =========================================================================

def discrete_curvature(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_triples: int = 200,
    seed: int = 42,
    epsilon: float = 1e-3,
    positions: Optional[List[int]] = None,
) -> Dict[str, object]:
    """
    Compute discrete Riemann curvature via composition non-additivity.

    Curvature = failure of per-token contributions to add linearly.
    For a triple (a, b, c) with a < b < c, compare:

        v_ab   = response at c from perturbing just at a  (T[c,a] @ h[a])
        v_bc   = response at c from perturbing just at b  (T[c,b] @ h[b])
        v_both = response at c from perturbing at a AND b simultaneously

    For a linear (flat) system: v_both = v_ab + v_bc  (superposition).
    Curvature = ||v_both - (v_ab + v_bc)|| / ||v_both||

    This measures the nonlinear interaction (cross-term) when multiple
    tokens are perturbed simultaneously. For idioms, the tokens interact
    non-additively (meaning is holistic), so curvature should be higher.

    This is computed at each layer separately (each = one VFE step).

    If positions is provided, only sample triples from those token positions
    (phrase-localized measurement). The full forward pass still uses all
    tokens for context.

    Returns dict with:
        'curvature_mean':      float, mean curvature across layers and triples
        'curvature_std':       float
        'curvature_max':       float
        'curvature_all':       (n_triples,) array (mean across layers)
        'curvature_per_layer': list of float, mean per layer
        'curvature_layer_all': (n_layers, n_triples) array
        'triples':             list of (a,b,c) triples
        'n_tokens':            int
        'd_model':             int
        'n_layers':            int
    """
    device = input_ids.device
    N = input_ids.shape[1]

    transformer_blocks = model.h if hasattr(model, 'h') else model.transformer.h
    n_layers = len(transformer_blocks)

    # Full forward to get hidden states
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    hidden_states = outputs.hidden_states
    d_model = hidden_states[0].shape[-1]

    triples = _sample_ordered_triples(N, max_triples=max_triples, seed=seed,
                                       positions=positions)
    n_tri = len(triples)
    if n_tri == 0:
        return {
            'curvature_mean': float('nan'), 'curvature_std': float('nan'),
            'curvature_max': float('nan'), 'curvature_all': np.zeros(0),
            'curvature_per_layer': [], 'curvature_layer_all': np.zeros((n_layers, 0)),
            'triples': [], 'n_tokens': N, 'd_model': d_model, 'n_layers': n_layers,
        }

    curvature_per_layer = np.zeros((n_layers, n_tri))

    for l in range(n_layers):
        h_in = hidden_states[l].detach().clone()
        block = transformer_blocks[l]

        with torch.no_grad():
            clean_out = block(h_in)[0]  # (1, N, d)

        # Collect unique positions to perturb
        unique_a = sorted(set(t[0] for t in triples))
        unique_b = sorted(set(t[1] for t in triples))
        unique_ab = sorted(set((t[0], t[1]) for t in triples))

        # JVP: perturb at position p with h_in[0,p], read output
        def jvp_at(pos):
            probe = h_in[0, pos]
            pn = probe.norm()
            if pn < 1e-30:
                return torch.zeros_like(clean_out[0])
            perturbed = h_in.clone()
            perturbed[0, pos] += epsilon * probe / pn
            with torch.no_grad():
                out = block(perturbed)[0]
            return ((out[0] - clean_out[0]) / epsilon) * pn  # (N, d)

        # JVP: perturb at both a and b simultaneously
        def jvp_both(a, b):
            probe_a = h_in[0, a]
            probe_b = h_in[0, b]
            na = probe_a.norm()
            nb = probe_b.norm()
            if na < 1e-30 or nb < 1e-30:
                return torch.zeros_like(clean_out[0])
            perturbed = h_in.clone()
            perturbed[0, a] += epsilon * probe_a / na
            perturbed[0, b] += epsilon * probe_b / nb
            with torch.no_grad():
                out = block(perturbed)[0]
            # Remove individual linear effects to isolate interaction
            return out[0] - clean_out[0]  # (N, d) — raw perturbation response

        # Cache individual JVPs
        jvp_cache = {}
        for pos in sorted(set(unique_a) | set(unique_b)):
            jvp_cache[pos] = jvp_at(pos)

        # For each triple: measure non-additivity at position c
        for t, (a, b, c) in enumerate(triples):
            resp_a = jvp_cache[a]  # response from perturbing a
            resp_b = jvp_cache[b]  # response from perturbing b

            # Linear prediction at c
            v_linear_c = resp_a[c] + resp_b[c]  # (d,) superposition

            # Actual joint response
            probe_a = h_in[0, a]
            probe_b = h_in[0, b]
            na = probe_a.norm()
            nb = probe_b.norm()
            if na < 1e-30 or nb < 1e-30:
                curvature_per_layer[l, t] = float('nan')
                continue

            perturbed = h_in.clone()
            perturbed[0, a] += epsilon * probe_a / na
            perturbed[0, b] += epsilon * probe_b / nb
            with torch.no_grad():
                out = block(perturbed)[0]
            v_actual_c = (out[0, c] - clean_out[0, c])  # (d,) raw response at c

            # Linear prediction (scaled consistently)
            v_pred_c = (resp_a[c] * epsilon / na + resp_b[c] * epsilon / nb)

            # Non-additivity
            diff = v_actual_c - v_pred_c
            scale = v_actual_c.norm().item()
            if scale < 1e-30:
                curvature_per_layer[l, t] = float('nan')
                continue
            curvature_per_layer[l, t] = diff.norm().item() / scale

    # Aggregates
    mean_per_triple = np.nanmean(curvature_per_layer, axis=0)
    finite = np.isfinite(mean_per_triple)
    clean = mean_per_triple[finite]
    layer_means = [float(np.nanmean(curvature_per_layer[l])) for l in range(n_layers)]

    return {
        'curvature_mean': float(np.mean(clean)) if len(clean) > 0 else float('nan'),
        'curvature_std': float(np.std(clean)) if len(clean) > 0 else float('nan'),
        'curvature_max': float(np.max(clean)) if len(clean) > 0 else float('nan'),
        'curvature_all': mean_per_triple,
        'curvature_per_layer': layer_means,
        'curvature_layer_all': curvature_per_layer,
        'triples': triples,
        'n_tokens': N,
        'd_model': d_model,
        'n_layers': n_layers,
    }


def load_model(model_name: str = 'gpt2', device: str = 'cpu'):
    """
    Load a pretrained transformer model for transport extraction.

    Args:
        model_name: HuggingFace model name (default: 'gpt2')
        device: 'cpu' or 'cuda'

    Returns:
        (model, tokenizer) tuple
    """
    from transformers import GPT2Model, GPT2Config

    # Try loading pretrained; fall back to random-init
    model = None
    for attempt_kwargs in [
        {'attn_implementation': 'eager'},  # newer transformers
        {},                                 # older transformers
    ]:
        if model is not None:
            break
        try:
            model = GPT2Model.from_pretrained(model_name, **attempt_kwargs)
        except (TypeError, ValueError):
            continue
        except (OSError, Exception):
            break  # network / cache error — go to fallback

    if model is None:
        import warnings
        warnings.warn(
            f"Could not load pretrained weights for '{model_name}'. "
            "Using random-initialized GPT-2 (same architecture). "
            "Geometric measurements are still valid for testing the framework."
        )
        config = GPT2Config()
        model = GPT2Model(config)

    # Ensure eager attention so we can extract attention weights.
    # Model init may silently default to sdpa which blocks output_attentions.
    for attr in ('_attn_implementation_internal', '_attn_implementation'):
        if hasattr(model.config, attr):
            setattr(model.config, attr, 'eager')
    model.config.output_attentions = True
    model = model.to(device)
    model.eval()

    # Try loading HF tokenizer; fall back to simple word tokenizer
    tokenizer = _get_tokenizer(model_name)

    return model, tokenizer


class _SimpleWordTokenizer:
    """
    Fallback word-level tokenizer when HuggingFace tokenizer is unavailable.

    Splits on whitespace and punctuation. Produces token IDs in [0, vocab_size).
    For geometric measurements, exact subword boundaries don't matter —
    what matters is that tokens are consistent across sentences.
    """
    def __init__(self, vocab_size: int = 50257):
        self._vocab_size = vocab_size
        self._vocab = {}
        self._next_id = 0

    @property
    def vocab_size(self):
        return self._vocab_size

    def _tokenize_text(self, text: str) -> List[str]:
        import re
        # Split into words and punctuation (similar to GPT-2 BPE pre-tokenization)
        return re.findall(r"\w+|[^\w\s]", text)

    def encode(self, text: str) -> List[int]:
        tokens = self._tokenize_text(text)
        ids = []
        for tok in tokens:
            if tok not in self._vocab:
                self._vocab[tok] = self._next_id % self._vocab_size
                self._next_id += 1
            ids.append(self._vocab[tok])
        return ids

    def decode(self, ids: List[int]) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        return ' '.join(inv.get(i, f'<{i}>') for i in ids)

    def __call__(self, text: str, **kwargs):
        ids = self.encode(text)
        return {'input_ids': ids, 'attention_mask': [1] * len(ids)}


def _get_tokenizer(model_name: str):
    """Try HF tokenizer, fall back to simple word tokenizer."""
    try:
        import os
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        # Verify it actually works
        test = tok.encode("test")
        if len(test) > 0:
            return tok
    except (ImportError, OSError, ValueError):
        pass

    try:
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained(model_name)
        test = tok.encode("test")
        if len(test) > 0:
            return tok
    except (ImportError, OSError, ValueError):
        pass

    import warnings
    warnings.warn(
        "Could not load HuggingFace tokenizer. Using simple word-level tokenizer. "
        "Token boundaries will differ from BPE but geometric measurements remain valid."
    )
    return _SimpleWordTokenizer()
