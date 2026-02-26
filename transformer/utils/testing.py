#!/usr/bin/env python
"""
Deep diagnostic for uniform attention rows.
Run: python debug_attention_uniform.py
"""

import torch
import numpy as np
torch.set_grad_enabled(False)
np.set_printoptions(precision=4, suppress=True, linewidth=120)

from transformer.core.model import GaugeTransformerLM


def diagnose():
    print("=" * 80)
    print("DEEP ATTENTION UNIFORMITY DIAGNOSTIC")
    print("=" * 80)

    # Create model with settings the user claims to have
    # embed_dim = 32 = 8*1 + 4*3 + 2*5 = 8 + 12 + 10 = 30... let's use 64
    # embed_dim = 64 = 16*1 + 8*3 + 4*5 = 16 + 24 + 20 = 60... pad to 64
    config = {
        'vocab_size': 100,
        'embed_dim': 63,
        'n_layers': 1,
        'hidden_dim': 128,
        'max_seq_len': 16,
        'kappa_beta': 0.1,
        'dropout': 0.0,
        'irrep_spec': [('l0', 15, 1), ('l1', 8, 3), ('l2', 4, 5)],  # Total: 16+24+20=60, padded to 64

        # The "fixes" that should help
        'mask_self_attention': True,
        'evolve_sigma': True,
        'alibi_slope': None,
        'gauge_fixed_priors': False,  # THIS IS THE ISSUE!

        # Disable other position sources for clarity
        'use_positional_embedding': False,
        'pos_encoding_mode': 'none',
        'use_identity_transport': False,  # Use actual transport
    }

    print("\n[CONFIG]")
    for k in ['mask_self_attention', 'evolve_sigma', 'alibi_slope',
              'gauge_fixed_priors', 'use_identity_transport', 'kappa_beta']:
        print(f"  {k}: {config.get(k)}")

    model = GaugeTransformerLM(config)
    model.eval()

    # Simple test sequence
    token_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    B, N = token_ids.shape
    K = config['embed_dim']  # 64

    print(f"\n[INPUT] Sequence shape: {token_ids.shape}, K={K}")

    # Get embeddings
    mu, sigma, phi = model.token_embed(token_ids)
    print(f"\n[EMBEDDINGS]")
    print(f"  mu shape: {mu.shape}, range: [{mu.min():.4f}, {mu.max():.4f}]")
    print(f"  sigma shape: {sigma.shape}")
    print(f"  phi shape: {phi.shape}, range: [{phi.min():.4f}, {phi.max():.4f}]")

    # Check if phi values are diverse
    phi_np = phi[0].numpy()
    print(f"\n[PHI VALUES] (should be diverse, not all similar)")
    for i in range(min(4, N)):
        print(f"  phi[{i}]: {phi_np[i]}")

    # Check mu diversity
    mu_np = mu[0].numpy()
    print(f"\n[MU NORMS] (should vary across tokens)")
    for i in range(min(4, N)):
        print(f"  ||mu[{i}]||: {np.linalg.norm(mu_np[i]):.4f}")

    # Check pairwise mu differences
    print(f"\n[MU PAIRWISE DISTANCES] (should vary!)")
    for i in range(min(3, N)):
        for j in range(min(3, N)):
            diff = np.linalg.norm(mu_np[i] - mu_np[j])
            print(f"  ||mu[{i}] - mu[{j}]||: {diff:.4f}")

    # Now compute transport operators manually
    generators = model.generators.numpy()
    print(f"\n[GENERATORS] shape: {generators.shape}")

    # Compute Omega_ij for a few pairs
    print(f"\n[TRANSPORT OPERATORS]")
    for i in range(min(2, N)):
        for j in range(min(3, N)):
            phi_i = phi_np[i]
            phi_j = phi_np[j]
            # phi_matrix = sum_a phi[a] * G[a]
            phi_mat_i = np.einsum('a,aij->ij', phi_i, generators)
            phi_mat_j = np.einsum('a,aij->ij', phi_j, generators)
            exp_phi_i = torch.linalg.matrix_exp(torch.from_numpy(phi_mat_i)).numpy()
            exp_neg_phi_j = torch.linalg.matrix_exp(torch.from_numpy(-phi_mat_j)).numpy()
            Omega_ij = exp_phi_i @ exp_neg_phi_j

            # Check if Omega is close to identity
            I = np.eye(K)
            dist_to_I = np.linalg.norm(Omega_ij - I, 'fro')
            print(f"  ||Omega[{i},{j}] - I||_F: {dist_to_I:.4f}")

    # Compute transported means
    print(f"\n[TRANSPORTED MEANS vs QUERY MEANS]")
    print("  (If gauge_fixed_priors=True and using same phi, these should be EQUAL!)")
    for i in range(min(3, N)):
        for j in range(min(3, N)):
            phi_i = phi_np[i]
            phi_j = phi_np[j]
            phi_mat_i = np.einsum('a,aij->ij', phi_i, generators)
            phi_mat_j = np.einsum('a,aij->ij', phi_j, generators)
            exp_phi_i = torch.linalg.matrix_exp(torch.from_numpy(phi_mat_i)).numpy()
            exp_neg_phi_j = torch.linalg.matrix_exp(torch.from_numpy(-phi_mat_j)).numpy()
            Omega_ij = exp_phi_i @ exp_neg_phi_j

            mu_j_transported = Omega_ij @ mu_np[j]
            diff = np.linalg.norm(mu_j_transported - mu_np[i])
            print(f"  ||Omega[{i},{j}] @ mu[{j}] - mu[{i}]||: {diff:.6f}")

    # Now run the actual forward pass and check attention
    print(f"\n[FORWARD PASS - ATTENTION WEIGHTS]")
    logits, attn_info = model.forward_with_attention(token_ids)
    beta = attn_info['beta']  # (B, n_heads, N, N)
    kl = attn_info.get('kl_matrix')

    print(f"  beta shape: {beta.shape}")

    # Average over heads
    beta_avg = beta.mean(dim=1)[0].numpy()  # (N, N)

    print(f"\n[ATTENTION MATRIX] (averaged over heads)")
    print("  (Causal mask applied - upper triangle should be 0)")
    print(beta_avg)

    # Check row uniformity
    print(f"\n[ROW UNIFORMITY CHECK]")
    for i in range(N):
        row = beta_avg[i, :i+1]  # Only visible positions
        if len(row) > 1:
            std = np.std(row)
            mean = np.mean(row)
            cv = std / mean if mean > 0 else 0
            print(f"  Row {i}: visible elements = {len(row)}, std = {std:.4f}, CV = {cv:.4f}")
            if cv < 0.1:
                print(f"         ^ UNIFORM (CV < 0.1)")

    # Check the KL matrix if available
    if kl is not None:
        kl_np = kl[0].numpy() if kl.dim() > 2 else kl.numpy()
        print(f"\n[KL MATRIX] (raw divergences, before -kl/kappa)")
        print(kl_np[:4, :4])  # First 4x4

        print(f"\n[KL STATISTICS]")
        # Exclude diagonal (which should be 0)
        mask = ~np.eye(N, dtype=bool)
        kl_off_diag = kl_np[mask]
        print(f"  Diagonal (should be 0): {np.diag(kl_np)[:4]}")
        print(f"  Off-diagonal range: [{kl_off_diag.min():.4f}, {kl_off_diag.max():.4f}]")
        print(f"  Off-diagonal mean: {kl_off_diag.mean():.4f}")
        print(f"  Off-diagonal std: {kl_off_diag.std():.4f}")

        if kl_off_diag.std() < 0.01:
            print(f"\n  *** ALL OFF-DIAGONAL KL VALUES ARE NEARLY IDENTICAL ***")
            print(f"  This is why attention is uniform!")

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)

    # KEY INSIGHT check
    print("\n[KEY INSIGHT]")
    print("With gauge_fixed_priors=True:")
    print("  - mu_i = R_i @ mu_base  where R_i = exp(phi_i @ G)")
    print("  - Omega_ij = R_i @ R_j^{-1}")
    print("  - Omega_ij @ mu_j = R_i @ R_j^{-1} @ R_j @ mu_base = R_i @ mu_base = mu_i")
    print("  - Therefore KL(q_i || Omega_ij[q_j]) = KL(q_i || q_i) = 0 for ALL pairs!")
    print("")
    print("gauge_fixed_priors=True makes attention DEGENERATE by design!")
    print("The gauge covariance property is too strong - it eliminates all information.")


if __name__ == "__main__":
    diagnose()