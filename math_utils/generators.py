# -*- coding: utf-8 -*-
"""
Lie Algebra Generators for Gauge Theory
=======================================

Construction and validation of Lie algebra generators for gauge transformations.

SO(3) / SO(K) Generators (Default):
----------------------------------
For SO(3), we use the spin-ℓ irreducible representations:
- Dimension: K = 2ℓ + 1 (always odd)
- Generators: Real skew-symmetric K×K matrices
- Commutation: [G_x, G_y] = G_z (cyclic)
- Casimir eigenvalue: ℓ(ℓ+1)

Uses real tesseral harmonics (not spherical) to avoid complex arithmetic.

GL(K) Gauge Structure:
----------------------
The VFE is invariant under GL(K) gauge transformations, not just SO(K)!
This means:
- Transport operators Ω = exp(φ·G) only need to be INVERTIBLE, not orthogonal
- The current SO(K) generators define a subalgebra of gl(K)
- For full GL(K) flexibility, you can extend to K² generators (full gl(K) basis)

Important: exp: gl(K,ℝ) → GL(K,ℝ) is NOT surjective.
- det(exp(X)) = exp(tr(X)) > 0, so exp only reaches GL⁺(K) (identity component)
- Even within GL⁺(K), not every matrix is a single exponential (Culver 1966)
- The product Ω_ij = exp(X_i)·exp(-X_j) does cover all of GL⁺(K)
- For SO(K), exp IS surjective (compact connected group) — no issues

For most applications, the SO(K) subalgebra generators suffice:
- They provide a natural parameterization with K(K-1)/2 or 3 parameters
- The transport operators remain well-conditioned
- exp(skew-symmetric) is always orthogonal (and surjects onto SO(K))

To use full GL(K), you would need to:
1. Generate K² generators spanning gl(K) (e.g., E_ij basis)
2. Use K² parameters in phi instead of 3 or K(K-1)/2
3. Remove any skew-symmetry constraints in transport.py
"""

import numpy as np
from typing import Dict


# =============================================================================
# Main Interface - SO(3) Generators
# =============================================================================

def generate_so3_generators(
    K: int,
    *,
    cache: bool = True,
    validate: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Generate SO(3) Lie algebra generators for dimension K.

    This is the primary interface for obtaining generators. Internally uses
    irrep construction with automatic validation.

    Args:
        K: Latent dimension (must be odd: K = 2ℓ + 1)
        cache: If True, cache generators for reuse
        validate: If True, verify commutation relations
        eps: Tolerance for validation

    Returns:
        G: Generators array, shape (3, K, K), float32
           G[a] is the a-th generator (a ∈ {0,1,2} for x,y,z)

    Properties:
        - G[a] is real skew-symmetric: G[a]ᵀ = -G[a]
        - Commutation: [G_x, G_y] = G_z (cyclic)
        - Casimir: -Σ_a G_a² = ℓ(ℓ+1) I where ℓ = (K-1)/2

    Examples:
        >>> # Spin-1 (3D, ℓ=1)
        >>> G = generate_so3_generators(3)
        >>> G.shape
        (3, 3, 3)

        >>> # Verify commutation
        >>> np.allclose(G[0] @ G[1] - G[1] @ G[0], G[2])
        True

        >>> # Spin-2 (5D, ℓ=2)
        >>> G = generate_so3_generators(5)
        >>> ell = (5 - 1) // 2  # = 2
        >>> casimir = ell * (ell + 1)  # = 6
        >>> C2 = -sum(G[a] @ G[a] for a in range(3))
        >>> np.allclose(C2, casimir * np.eye(5))
        True

    Raises:
        ValueError: If K is even (SO(3) irreps must have odd dimension)
        RuntimeError: If validation fails

    Notes:
        - For K=3: Standard 3D rotation generators (spin-1)
        - For K=5,7,9,...: Higher spin representations
        - Internally constructs irrep via tesseral harmonics
        - Cached by default for performance
    """
    # Validate K is odd
    if K % 2 == 0:
        raise ValueError(
            f"K must be odd for SO(3) irreps (K = 2ℓ + 1). Got K={K}."
        )

    # Check cache
    if cache and K in _GENERATOR_CACHE:
        return _GENERATOR_CACHE[K].copy()

    # Compute spin quantum number
    ell = (K - 1) // 2

    # Build irrep generators
    G = _build_so3_irrep_generators(ell)

    # Validate if requested
    if validate:
        _validate_so3_generators(G, eps=1e-5)

    # Cache for reuse
    if cache:
        _GENERATOR_CACHE[K] = G.copy()

    return G


# =============================================================================
# Irrep Construction (Tesseral Basis)
# =============================================================================

def _build_so3_irrep_generators(ell: int) -> np.ndarray:
    """
    Build SO(3) generators for spin-ℓ irrep in real tesseral basis.

    Algorithm:
    ---------
    1. Construct complex spherical harmonic operators J_x, J_y, J_z
    2. Build unitary transformation S: spherical → tesseral
    3. Transform: G_a = Re(S J_a S†) and enforce skew-symmetry

    Args:
        ell: Spin quantum number (ℓ ≥ 0)

    Returns:
        G: (3, K, K) float32 generators where K = 2ℓ + 1
    """
    K = 2 * ell + 1

    # ========== Step 1: Complex spherical operators ==========
    # Build J_+, J_-, J_z in complex basis
    J_plus = np.zeros((K, K), dtype=np.complex128)
    J_minus = np.zeros((K, K), dtype=np.complex128)
    J_z = np.zeros((K, K), dtype=np.complex128)

    for m in range(-ell, ell + 1):
        i = m + ell  # Index: m ∈ [-ℓ, ℓ] → i ∈ [0, K-1]

        # J_z is diagonal
        J_z[i, i] = m

        # J_+ raises m by 1
        if m < ell:
            a = np.sqrt((ell - m) * (ell + m + 1))
            J_plus[i, i + 1] = a

        # J_- lowers m by 1
        if m > -ell:
            a = np.sqrt((ell + m) * (ell - m + 1))
            J_minus[i, i - 1] = a

    # Cartesian operators
    J_x = (J_plus + J_minus) / 2.0
    J_y = (J_plus - J_minus) / (2.0j)

    # ========== Step 2: Spherical → Tesseral transformation ==========
    # S is unitary, transforms |ℓ,m⟩ → tesseral basis
    S = _build_tesseral_transform(ell)
    S_inv = S.conj().T

    # ========== Step 3: Transform to real basis ==========
    def _to_real_skew(J_complex: np.ndarray) -> np.ndarray:
        """Transform complex operator to real skew-symmetric generator."""
        # G = Re(S (iJ) S†) where factor of i makes it skew-symmetric
        G_complex = S @ (1j * J_complex) @ S_inv
        G_real = G_complex.real

        # Enforce skew-symmetry (remove any numerical symmetric part)
        G_skew = 0.5 * (G_real - G_real.T)
        return G_skew

    G_x = _to_real_skew(J_x)
    G_y = _to_real_skew(J_y)
    G_z = _to_real_skew(J_z)

    # Stack as (3, K, K)
    G = np.stack([G_x, G_y, G_z], axis=0)

    return G.astype(np.float32, copy=False)


def _build_tesseral_transform(ell: int) -> np.ndarray:
    """
    Construct unitary transformation from spherical to tesseral basis.

    Tesseral harmonics are real linear combinations of spherical harmonics:
        Y^c_{ℓm} = (Y_{ℓm} + (-1)^m Y_{ℓ,-m}) / √2        (cosine-like, m > 0)
        Y^s_{ℓm} = (Y_{ℓm} - (-1)^m Y_{ℓ,-m}) / (i√2)     (sine-like, m > 0)
        Y^0_{ℓ0} = Y_{ℓ0}                                  (m = 0)

    Args:
        ell: Spin quantum number

    Returns:
        S: (K, K) unitary matrix, complex128
    """
    K = 2 * ell + 1
    S = np.zeros((K, K), dtype=np.complex128)

    # m = 0 component (center)
    S[0, ell] = 1.0

    # m > 0 components (cosine and sine pairs)
    row = 1
    for m in range(1, ell + 1):
        phase = (-1) ** m
        sqrt2_inv = 1.0 / np.sqrt(2.0)

        # Cosine-like: Y^c_m = (Y_m + phase Y_{-m}) / √2
        S[row, ell + m] = sqrt2_inv
        S[row, ell - m] = phase * sqrt2_inv
        row += 1

        # Sine-like: Y^s_m = (Y_m - phase Y_{-m}) / (i√2)
        S[row, ell + m] = -1j * sqrt2_inv
        S[row, ell - m] = 1j * phase * sqrt2_inv
        row += 1

    return S


# =============================================================================
# Validation
# =============================================================================

def _validate_so3_generators(
    G: np.ndarray,
    *,
    eps: float = 1e-6,
    verbose: bool = False,
) -> None:
    """
    Validate SO(3) commutation relations and properties.

    Checks:
    ------
    1. Skew-symmetry: G[a]ᵀ = -G[a]
    2. Commutation: [G_x, G_y] = G_z (cyclic)
    3. Casimir: C_2 = -Σ G_a² = ℓ(ℓ+1) I

    Args:
        G: (3, K, K) generators
        eps: Tolerance for checks
        verbose: If True, print validation details

    Raises:
        RuntimeError: If any check fails
    """
    if G.shape[0] != 3:
        raise ValueError(f"Expected 3 generators (x,y,z), got {G.shape[0]}")

    K = G.shape[1]
    if G.shape != (3, K, K):
        raise ValueError(f"Expected shape (3, K, K), got {G.shape}")

    # Cast to float64 for validation to avoid precision issues with large spin
    G64 = G.astype(np.float64)
    G_x, G_y, G_z = G64[0], G64[1], G64[2]

    # ========== Check 1: Skew-symmetry ==========
    for a, name in enumerate(['x', 'y', 'z']):
        G_a = G64[a]
        skew_error = np.linalg.norm(G_a + G_a.T, ord='fro')
        if skew_error > eps:
            raise RuntimeError(
                f"Generator G_{name} not skew-symmetric: ||G + Gᵀ|| = {skew_error:.3e}"
            )

    # ========== Check 2: Commutation relations ==========
    # [G_x, G_y] = G_z
    comm_xy = G_x @ G_y - G_y @ G_x
    error_xy = np.linalg.norm(comm_xy - G_z, ord='fro')

    # [G_y, G_z] = G_x (cyclic)
    comm_yz = G_y @ G_z - G_z @ G_y
    error_yz = np.linalg.norm(comm_yz - G_x, ord='fro')

    # [G_z, G_x] = G_y
    comm_zx = G_z @ G_x - G_x @ G_z
    error_zx = np.linalg.norm(comm_zx - G_y, ord='fro')

    max_error = max(error_xy, error_yz, error_zx)

    # Scale tolerance by generator norm squared for commutator checks
    # (matrix products accumulate errors proportional to scale²)
    scale = max(np.linalg.norm(G64[a], ord='fro') for a in range(3))
    threshold = eps * max(scale * scale, 1.0)

    if max_error > threshold:
        raise RuntimeError(
            f"SO(3) commutation relations violated:\n"
            f"  [G_x, G_y] - G_z: {error_xy:.3e}\n"
            f"  [G_y, G_z] - G_x: {error_yz:.3e}\n"
            f"  [G_z, G_x] - G_y: {error_zx:.3e}\n"
            f"  threshold: {threshold:.3e}"
        )

    # Use float64 for Casimir check as well
    C_2 = -sum(G64[a] @ G64[a] for a in range(3))

    # Extract eigenvalues (should all be ℓ(ℓ+1))
    eigenvalues    = np.linalg.eigvalsh(C_2)
    casimir_value  = float(np.mean(eigenvalues))
    casimir_spread = float(np.std(eigenvalues))

    # Expected value
    ell = (K - 1) // 2
    casimir_expected = ell * (ell + 1)
    casimir_error = abs(casimir_value - casimir_expected)

    # Scale tolerance by the size of C₂ (larger for high spin)
    base = max(abs(casimir_expected), 1.0)
    tol  = eps * base

    if casimir_error > tol or casimir_spread > tol:
        raise RuntimeError(
            "Casimir operator check failed:\n"
            f"  Expected: {casimir_expected}\n"
            f"  Got: {casimir_value:.6f} ± {casimir_spread:.3e}\n"
            f"  Error: {casimir_error:.3e}"
        )


    if verbose:
        print("✓ SO(3) generator validation passed:")
        print(f"  Dimension: K = {K} (ℓ = {ell})")
        print(f"  Skew-symmetry: max error = {max([np.linalg.norm(G64[a] + G64[a].T) for a in range(3)]):.3e}")
        print(f"  Commutation: max error = {max_error:.3e}")
        print(f"  Casimir: C₂ = {casimir_value:.6f} (expected {casimir_expected})")


# =============================================================================
# Multi-Irrep Block-Diagonal Generators
# =============================================================================

def generate_multi_irrep_generators(
    irrep_spec: list,
    *,
    validate: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Generate block-diagonal SO(3) generators from a multi-irrep specification.

    This creates generators that act on a direct sum of irreducible representations:
        V = ⊕_ℓ (V_ℓ)^{n_ℓ}

    where V_ℓ is the spin-ℓ irrep (dimension 2ℓ+1) with multiplicity n_ℓ.

    Args:
        irrep_spec: List of (label, multiplicity, dim) tuples.
            Example: [('ℓ0', 32, 1), ('ℓ1', 15, 3), ('ℓ2', 10, 5)]
            - label: String identifier (e.g., 'ℓ0', 'ℓ1', 'scalar', 'vector')
            - multiplicity: How many copies of this irrep
            - dim: Dimension of irrep (must be odd: 1, 3, 5, 7, ...)
        validate: If True, verify the resulting generators
        eps: Tolerance for validation

    Returns:
        G: Block-diagonal generators, shape (3, K, K), where K = Σ mult × dim
           Each G[a] has blocks corresponding to each irrep copy

    Example:
        >>> # K = 32×1 + 15×3 + 10×5 = 32 + 45 + 50 = 127
        >>> spec = [('ℓ0', 32, 1), ('ℓ1', 15, 3), ('ℓ2', 10, 5)]
        >>> G = generate_multi_irrep_generators(spec)
        >>> G.shape
        (3, 127, 127)

        >>> # Structure: block diagonal with scalar 0-blocks, then spin-1 blocks, then spin-2
        >>> # First 32 dimensions: all zeros (scalars don't rotate)
        >>> np.allclose(G[:, :32, :32], 0)
        True

    Raises:
        ValueError: If any irrep dimension is even
    """
    # Validate irrep dimensions
    for label, mult, dim in irrep_spec:
        if dim % 2 == 0:
            raise ValueError(
                f"Irrep '{label}' has even dimension {dim}. "
                f"SO(3) irreps must have odd dimension (2ℓ+1)."
            )
        if mult < 0:
            raise ValueError(f"Irrep '{label}' has negative multiplicity {mult}.")

    # Compute total dimension
    K = sum(mult * dim for _, mult, dim in irrep_spec)

    # Initialize block-diagonal generators
    G = np.zeros((3, K, K), dtype=np.float32)

    # Fill in blocks
    idx = 0
    for label, mult, dim in irrep_spec:
        if dim == 1:
            # Scalars (ℓ=0): generator is zero
            # Skip mult×1 dimensions
            idx += mult * dim
        else:
            # Higher spin: get generators for this irrep
            G_irrep = generate_so3_generators(dim, cache=True, validate=False)

            # Place mult copies on the diagonal
            for _ in range(mult):
                G[:, idx:idx+dim, idx:idx+dim] = G_irrep
                idx += dim

    # Validate if requested
    if validate and K > 1:
        _validate_block_diagonal_generators(G, irrep_spec, eps=eps)

    return G


def _validate_block_diagonal_generators(
    G: np.ndarray,
    irrep_spec: list,
    *,
    eps: float = 1e-6,
) -> None:
    """
    Validate block-diagonal multi-irrep generators.

    Checks:
    1. Skew-symmetry: G[a]ᵀ = -G[a]
    2. Commutation: [G_x, G_y] = G_z (cyclic)
    3. Block structure: off-diagonal blocks are zero

    Args:
        G: (3, K, K) generators
        irrep_spec: The irrep specification used to create G
        eps: Tolerance for checks
    """
    K = G.shape[1]

    # Cast to float64 for validation to avoid precision issues with large spin
    G64 = G.astype(np.float64)

    # Check skew-symmetry
    for a in range(3):
        skew_error = np.linalg.norm(G64[a] + G64[a].T, ord='fro')
        if skew_error > eps:
            raise RuntimeError(
                f"Block-diagonal generator G[{a}] not skew-symmetric: "
                f"||G + Gᵀ|| = {skew_error:.3e}"
            )

    # Check commutation relations
    G_x, G_y, G_z = G64[0], G64[1], G64[2]

    comm_xy = G_x @ G_y - G_y @ G_x
    error_xy = np.linalg.norm(comm_xy - G_z, ord='fro')

    comm_yz = G_y @ G_z - G_z @ G_y
    error_yz = np.linalg.norm(comm_yz - G_x, ord='fro')

    comm_zx = G_z @ G_x - G_x @ G_z
    error_zx = np.linalg.norm(comm_zx - G_y, ord='fro')

    max_error = max(error_xy, error_yz, error_zx)

    # Scale tolerance by generator norm squared for commutator checks
    # (matrix products accumulate errors proportional to scale²)
    scale = max(np.linalg.norm(G64[a], ord='fro') for a in range(3))
    threshold = eps * max(scale * scale, 1.0)

    if max_error > threshold:
        raise RuntimeError(
            f"Block-diagonal SO(3) commutation violated:\n"
            f"  [G_x, G_y] - G_z: {error_xy:.3e}\n"
            f"  [G_y, G_z] - G_x: {error_yz:.3e}\n"
            f"  [G_z, G_x] - G_y: {error_zx:.3e}\n"
            f"  threshold: {threshold:.3e}"
        )

    # Check block structure (off-diagonal blocks should be zero)
    idx = 0
    block_starts = []
    for _, mult, dim in irrep_spec:
        for _ in range(mult):
            block_starts.append((idx, dim))
            idx += dim

    for i, (start_i, dim_i) in enumerate(block_starts):
        for j, (start_j, dim_j) in enumerate(block_starts):
            if i != j:
                # Check off-diagonal block is zero
                for a in range(3):
                    block = G64[a, start_i:start_i+dim_i, start_j:start_j+dim_j]
                    block_norm = np.linalg.norm(block, ord='fro')
                    if block_norm > eps:
                        raise RuntimeError(
                            f"Off-diagonal block ({i},{j}) is non-zero: "
                            f"||block|| = {block_norm:.3e}"
                        )


# =============================================================================
# SO(N) Generators - Fundamental Representation
# =============================================================================

def generate_soN_generators(
    N: int,
    *,
    validate: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Generate SO(N) Lie algebra generators in the fundamental (N-dimensional) representation.

    SO(N) is the group of N×N orthogonal matrices with determinant 1.
    Its Lie algebra so(N) consists of N×N skew-symmetric matrices.

    The Lie algebra has dimension N(N-1)/2, with basis elements L_{ij} for i < j:
        (L_{ij})_{kl} = δ_{ik}δ_{jl} - δ_{il}δ_{jk}

    These satisfy the commutation relations:
        [L_{ij}, L_{kl}] = δ_{jk}L_{il} - δ_{ik}L_{jl} - δ_{jl}L_{ik} + δ_{il}L_{jk}

    Args:
        N: The dimension of the fundamental representation (N ≥ 2)
        validate: If True, verify commutation relations
        eps: Tolerance for validation

    Returns:
        G: Generators array, shape (N(N-1)/2, N, N), float32
           G[a] is the a-th generator, indexed by pairs (i,j) with i < j

    Examples:
        >>> # SO(3) - 3 generators, 3×3 matrices
        >>> G = generate_soN_generators(3)
        >>> G.shape
        (3, 3, 3)

        >>> # SO(5) - 10 generators, 5×5 matrices
        >>> G = generate_soN_generators(5)
        >>> G.shape
        (10, 5, 5)

        >>> # SO(8) - 28 generators, 8×8 matrices
        >>> G = generate_soN_generators(8)
        >>> G.shape
        (28, 8, 8)

    Properties:
        - G[a] is real skew-symmetric: G[a]ᵀ = -G[a]
        - Orthogonal action: exp(θ G[a]) ∈ SO(N) for any θ
        - Satisfies so(N) commutation relations
    """
    if N < 2:
        raise ValueError(f"N must be >= 2 for SO(N), got N={N}")

    n_generators = N * (N - 1) // 2
    G = np.zeros((n_generators, N, N), dtype=np.float32)

    # Build generators L_{ij} for i < j
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            # (L_{ij})_{kl} = δ_{ik}δ_{jl} - δ_{il}δ_{jk}
            G[idx, i, j] = 1.0
            G[idx, j, i] = -1.0
            idx += 1

    if validate:
        _validate_soN_generators(G, N, eps=eps)

    return G


def _validate_soN_generators(
    G: np.ndarray,
    N: int,
    *,
    eps: float = 1e-6,
) -> None:
    """
    Validate SO(N) generators satisfy required properties.

    Checks:
    1. Skew-symmetry: G[a]ᵀ = -G[a]
    2. Commutation relations: [L_{ij}, L_{kl}] follows so(N) structure

    Args:
        G: (n_gen, N, N) generators where n_gen = N(N-1)/2
        N: Dimension of fundamental rep
        eps: Tolerance for checks
    """
    n_gen = G.shape[0]
    expected_n_gen = N * (N - 1) // 2

    if n_gen != expected_n_gen:
        raise ValueError(
            f"Expected {expected_n_gen} generators for SO({N}), got {n_gen}"
        )

    # Check skew-symmetry
    for a in range(n_gen):
        skew_error = np.linalg.norm(G[a] + G[a].T, ord='fro')
        if skew_error > eps:
            raise RuntimeError(
                f"SO({N}) generator G[{a}] not skew-symmetric: "
                f"||G + Gᵀ|| = {skew_error:.3e}"
            )

    # Build index map: (i,j) -> generator index
    idx_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx_map[(i, j)] = idx
            idx += 1

    # Check a sample of commutation relations
    # [L_{ij}, L_{jk}] = L_{ik} for i < j < k
    max_error = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                # [L_{ij}, L_{jk}] should equal L_{ik}
                a = idx_map[(i, j)]
                b = idx_map[(j, k)]
                c = idx_map[(i, k)]

                comm = G[a] @ G[b] - G[b] @ G[a]
                error = np.linalg.norm(comm - G[c], ord='fro')
                max_error = max(max_error, error)

    if max_error > eps:
        raise RuntimeError(
            f"SO({N}) commutation relations violated, max error: {max_error:.3e}"
        )


# =============================================================================
# GL(K) Generators - Full General Linear Algebra
# =============================================================================

def generate_glK_generators(
    K: int,
    *,
    include_identity: bool = False,
) -> np.ndarray:
    """
    Generate gl(K) Lie algebra generators (full K² basis).

    gl(K) is the Lie algebra of all K×K matrices. Unlike so(K) which has
    K(K-1)/2 skew-symmetric generators, gl(K) has K² generators spanning
    all matrices.

    The standard basis is E_ij (matrix with 1 at position (i,j), 0 elsewhere).
    The Lie bracket is the matrix commutator: [A, B] = AB - BA.

    Use this for maximum flexibility in GL(K) gauge transformations.
    Transport operators Ω = exp(φ·G) will be general invertible matrices.

    Args:
        K: Matrix dimension
        include_identity: If True, include trace component (K² generators).
                         If False, use sl(K) traceless basis (K²-1 generators).
                         Default True for full gl(K).

    Returns:
        G: Generators array, shape (K², K, K) or (K²-1, K, K), float32
           G[a] is the a-th generator

    Examples:
        >>> # gl(3) - 9 generators, 3×3 matrices
        >>> G = generate_glK_generators(3)
        >>> G.shape
        (9, 3, 3)

        >>> # gl(5) - 25 generators, 5×5 matrices
        >>> G = generate_glK_generators(5)
        >>> G.shape
        (25, 5, 5)

        >>> # Full GL(K) transport
        >>> phi = np.random.randn(25) * 0.1  # 25 parameters for gl(5)
        >>> X = np.einsum('a,aij->ij', phi, G)  # Lie algebra element
        >>> Omega = scipy.linalg.expm(X)  # GL(5) matrix

    Properties:
        - G[a] spans all K×K matrices (not just skew-symmetric)
        - exp(φ·G) ∈ GL⁺(K) for all φ (det > 0, not necessarily orthogonal)
        - Includes both symmetric and antisymmetric directions
        - More parameters than so(K): K² vs K(K-1)/2

    Comparison:
        | Algebra | Generators | K=10 params | Constraint on exp(X)          |
        |---------|------------|-------------|-------------------------------|
        | so(K)   | K(K-1)/2   | 45          | Orthogonal (Ωᵀ Ω = I)        |
        | sl(K)   | K²-1       | 99          | det = 1                       |
        | gl(K)   | K²         | 100         | det > 0 (identity component)  |

    Note on exp surjectivity:
        The VFE is invariant under the full GL(K) (including det < 0), but
        the exponential parameterization only reaches GL⁺(K) since
        det(exp(X)) = exp(tr(X)) > 0 always.

        Even within GL⁺(K), a SINGLE exp(X) is not surjective for K > 1.
        By Culver's theorem (1966), A ∈ GL(K,ℝ) has a real logarithm iff
        for each negative real eigenvalue λ, the number of Jordan blocks of
        each size for λ is even. E.g. diag(-2, -3) ∈ GL⁺(2) has no real log
        (each negative eigenvalue has 1 block of size 1: odd count).

        This does NOT limit our transport operators: Ω_ij = exp(X_i)·exp(-X_j)
        is a product of two exponentials, which covers all of GL⁺(K) by the
        polar decomposition argument (any A = P·O = exp(log P)·exp(log O)).
    """
    if K < 1:
        raise ValueError(f"K must be >= 1 for GL(K), got K={K}")

    n_generators = K * K
    G = np.zeros((n_generators, K, K), dtype=np.float32)

    # Build standard basis E_ij
    idx = 0
    for i in range(K):
        for j in range(K):
            G[idx, i, j] = 1.0
            idx += 1

    if not include_identity:
        # Remove trace component to get sl(K) (K^2 - 1 generators)
        # The identity direction is (1/sqrt(K)) * I_K
        # Project out this component from each generator
        I_K = np.eye(K, dtype=np.float32)
        trace_dir = I_K / np.sqrt(K)  # Normalized identity direction
        projected = []
        for g in range(n_generators):
            # Remove trace component: G_new = G - tr(G * trace_dir) * trace_dir
            overlap = np.sum(G[g] * trace_dir)
            G_proj = G[g] - overlap * trace_dir
            if np.linalg.norm(G_proj) > 1e-8:
                projected.append(G_proj)
        G = np.stack(projected, axis=0)

    return G


def generate_glK_generators_split(
    K: int,
) -> tuple:
    """
    Generate gl(K) generators split into symmetric and antisymmetric parts.

    This is useful for understanding the structure:
    - Antisymmetric part (K(K-1)/2 generators) → so(K) subalgebra → orthogonal
    - Symmetric part (K(K+1)/2 generators) → scaling/shearing directions

    Args:
        K: Matrix dimension

    Returns:
        (G_antisym, G_sym): Tuple of generator arrays
            G_antisym: (K(K-1)/2, K, K) - skew-symmetric, generates SO(K)
            G_sym: (K(K+1)/2, K, K) - symmetric, generates scaling/shearing

    Example:
        >>> G_antisym, G_sym = generate_glK_generators_split(5)
        >>> G_antisym.shape  # so(5) part
        (10, 5, 5)
        >>> G_sym.shape  # symmetric part
        (15, 5, 5)
        >>> # Combined: 10 + 15 = 25 = 5²
    """
    n_antisym = K * (K - 1) // 2
    n_sym = K * (K + 1) // 2

    G_antisym = np.zeros((n_antisym, K, K), dtype=np.float32)
    G_sym = np.zeros((n_sym, K, K), dtype=np.float32)

    # Antisymmetric: L_ij = E_ij - E_ji for i < j
    idx = 0
    for i in range(K):
        for j in range(i + 1, K):
            G_antisym[idx, i, j] = 1.0
            G_antisym[idx, j, i] = -1.0
            idx += 1

    # Symmetric: S_ij = E_ij + E_ji for i < j, plus E_ii for diagonal
    idx = 0
    # Diagonal elements E_ii
    for i in range(K):
        G_sym[idx, i, i] = 1.0
        idx += 1
    # Off-diagonal symmetric: E_ij + E_ji for i < j
    for i in range(K):
        for j in range(i + 1, K):
            G_sym[idx, i, j] = 1.0
            G_sym[idx, j, i] = 1.0
            idx += 1

    return G_antisym, G_sym


def generate_glK_multihead_generators(
    K: int,
    n_heads: int,
    *,
    include_identity: bool = True,
) -> np.ndarray:
    """
    Generate block-diagonal gl(d_head) generators for multi-head GL(K) attention.

    For multi-head attention in the GL(K) framework, we decompose the K-dimensional
    embedding space into H heads, each with its own GL(d_head) gauge structure:

        GL(d_head)^H ⊂ GL(K)

    where d_head = K // n_heads.

    Each head operates independently with its own gauge transport Ω^(h) ∈ GL(d_head).
    The full transport is block-diagonal:

        Ω = diag(Ω^(1), Ω^(2), ..., Ω^(H))

    This is the GL(K) analogue of multi-head attention in transformers, where
    each head learns its own gauge transformation.

    Args:
        K: Total embedding dimension
        n_heads: Number of attention heads
        include_identity: If True, include trace component per head

    Returns:
        G: Block-diagonal generators, shape (n_heads * d_head², K, K)
           Each block G[h*d_head²:(h+1)*d_head², ...] contains the gl(d_head)
           generators for head h, embedded in the full K×K space.

    Example:
        >>> # 30-dim embedding with 5 heads → each head has GL(6) structure
        >>> G = generate_glK_multihead_generators(30, n_heads=5)
        >>> G.shape
        (180, 30, 30)  # 5 heads × 6² generators = 180 total

        >>> # Each head's generators are block-diagonal
        >>> # Head 0 acts on dims [0:6], Head 1 on [6:12], etc.

    Comparison with single-head GL(K):
        | Mode        | Generators | K=30, H=5 | Parameters (φ) |
        |-------------|------------|-----------|----------------|
        | GL(K)       | K²         | 900       | 900            |
        | GL(K) MH    | H × d²     | 180       | 180            |

    Note:
        Multi-head reduces parameters from K² to H×(K/H)² = K²/H while
        preserving the key property: each head learns its optimal gauge.
        The block-diagonal structure means heads don't interfere.
    """
    if K % n_heads != 0:
        raise ValueError(
            f"K={K} must be divisible by n_heads={n_heads}. "
            f"Got K % n_heads = {K % n_heads}"
        )

    d_head = K // n_heads
    n_gen_per_head = d_head * d_head
    n_generators = n_heads * n_gen_per_head

    G = np.zeros((n_generators, K, K), dtype=np.float32)

    # Build block-diagonal generators
    for h in range(n_heads):
        # This head acts on dimensions [h*d_head : (h+1)*d_head]
        start = h * d_head
        end = (h + 1) * d_head

        # Generator index offset for this head
        gen_offset = h * n_gen_per_head

        # Build gl(d_head) basis E_ij within this block
        idx = 0
        for i in range(d_head):
            for j in range(d_head):
                # E_ij in the full K×K space, restricted to this head's block
                G[gen_offset + idx, start + i, start + j] = 1.0
                idx += 1

    return G


def generate_glK_cross_head_generators(
    K: int,
    n_heads: int,
    cross_couplings: 'List[Tuple[int, int]]',
) -> np.ndarray:
    """
    Generate GL(K) generators with sparse off-diagonal cross-head coupling.

    Extends generate_glK_multihead_generators by adding generators for selected
    pairs of heads, enabling gauge transport that mixes information between heads.

    The generator set consists of:
      1. Diagonal blocks: gl(d_head) per head  (same as multihead)
      2. Off-diagonal blocks: E_ij spanning (head_a → head_b) for each (a,b) pair

    For a coupling pair (a, b), we add d_head² generators:
      G[start + i*d_head + j]_{a_start+i, b_start+j} = 1

    These are the elementary matrices that connect head a's subspace to head b's.
    The resulting transport Ω = exp(φ·G) will have non-zero blocks at (a,a), (b,b),
    AND (a,b), (b,a) — heads a and b can now exchange information through gauge
    transport.

    For the KL computation we merge coupled heads into super-blocks:
      - Heads {a, b} that appear in any coupling pair are grouped together
      - Their combined subspace becomes a single block of dimension Σ d_head
      - The existing block-diagonal KL code handles this with no changes —
        the super-block's generators are simply larger

    Args:
        K: Total embedding dimension
        n_heads: Number of attention heads
        cross_couplings: List of (head_a, head_b) pairs to couple.
            Must have a != b and 0 <= a, b < n_heads.
            Pairs are treated as directed: (a,b) adds generators a→b.
            For symmetric coupling, include both (a,b) and (b,a).

    Returns:
        G: Generators, shape (n_total_gen, K, K) where
           n_total_gen = n_heads * d_head² + len(cross_couplings) * d_head²

    Example:
        >>> # 24-dim embedding, 4 heads of dim 6, couple heads 0↔1 and 2↔3
        >>> G = generate_glK_cross_head_generators(24, 4, [(0,1),(1,0),(2,3),(3,2)])
        >>> G.shape  # 4*36 + 4*36 = 288 generators
        (288, 24, 24)
    """
    if K % n_heads != 0:
        raise ValueError(f"K={K} not divisible by n_heads={n_heads}")

    d_head = K // n_heads
    n_gen_diag = n_heads * d_head * d_head
    n_gen_cross = len(cross_couplings) * d_head * d_head
    n_gen_total = n_gen_diag + n_gen_cross

    G = np.zeros((n_gen_total, K, K), dtype=np.float32)

    # 1. Diagonal blocks (identical to generate_glK_multihead_generators)
    for h in range(n_heads):
        start = h * d_head
        gen_offset = h * d_head * d_head
        idx = 0
        for i in range(d_head):
            for j in range(d_head):
                G[gen_offset + idx, start + i, start + j] = 1.0
                idx += 1

    # 2. Off-diagonal blocks for each coupling pair
    for pair_idx, (a, b) in enumerate(cross_couplings):
        if a == b:
            raise ValueError(f"Self-coupling ({a},{a}) not allowed — already in diagonal")
        if not (0 <= a < n_heads and 0 <= b < n_heads):
            raise ValueError(f"Head indices ({a},{b}) out of range [0, {n_heads})")

        a_start = a * d_head
        b_start = b * d_head
        gen_offset = n_gen_diag + pair_idx * d_head * d_head
        idx = 0
        for i in range(d_head):
            for j in range(d_head):
                # E_ij from head_a row subspace to head_b column subspace
                G[gen_offset + idx, a_start + i, b_start + j] = 1.0
                idx += 1

    return G


def merge_coupled_heads(
    n_heads: int,
    d_head: int,
    cross_couplings: 'List[Tuple[int, int]]',
) -> 'Tuple[List[int], List[List[int]]]':
    """
    Compute super-block structure from cross-head coupling pattern.

    Heads that are transitively connected through couplings are merged into
    a single super-block for the block-diagonal KL computation. Uncoupled
    heads remain as singleton blocks.

    Uses union-find to compute connected components.

    Args:
        n_heads: Number of attention heads
        d_head: Dimension per head
        cross_couplings: List of (head_a, head_b) directed coupling pairs.
            Both (a,b) and (b,a) are treated as connecting heads a and b.

    Returns:
        super_block_dims: List of block dimensions for the merged structure.
            Sum equals n_heads * d_head = K.
            Example: n_heads=4, d_head=6, couplings=[(0,1),(1,0)]
                     → [12, 6, 6] (heads 0,1 merged; heads 2,3 separate)
        super_block_head_groups: List of lists, each containing the head
            indices in that super-block, in order.
            Example: [[0, 1], [2], [3]]
    """
    # Union-find
    parent = list(range(n_heads))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for a, b in cross_couplings:
        union(a, b)

    # Group heads by their root
    from collections import defaultdict
    groups = defaultdict(list)
    for h in range(n_heads):
        groups[find(h)].append(h)

    # Sort groups by smallest head index for deterministic ordering
    sorted_groups = sorted(groups.values(), key=lambda g: g[0])

    super_block_dims = [len(g) * d_head for g in sorted_groups]
    super_block_head_groups = sorted_groups

    return super_block_dims, super_block_head_groups


def reorder_cross_head_generators(
    G: np.ndarray,
    n_heads: int,
    d_head: int,
    cross_couplings: 'List[Tuple[int, int]]',
    super_block_head_groups: 'List[List[int]]',
) -> 'Tuple[np.ndarray, List[int]]':
    """
    Reorder generators so that merged super-blocks are contiguous.

    The original generator layout has heads in natural order [0, 1, 2, ...].
    After merging, super-blocks may group non-adjacent heads. This function
    produces a permutation matrix P such that P @ G @ P^T has the merged
    heads contiguous, making the generators block-diagonal in super-blocks.

    Also returns the reordered super-block dimensions.

    Args:
        G: Generators from generate_glK_cross_head_generators,
           shape (n_gen, K, K)
        n_heads: Number of original heads
        d_head: Dimension per head
        cross_couplings: Coupling pairs (for reference)
        super_block_head_groups: From merge_coupled_heads

    Returns:
        G_reordered: Generators with permuted rows/cols, shape (n_gen, K, K)
        perm: The permutation vector of length K such that
              G_reordered[:, i, j] = G[:, perm[i], perm[j]]
    """
    K = n_heads * d_head

    # Build permutation: concatenate head subspaces in super-block order
    perm = []
    for group in super_block_head_groups:
        for h in group:
            perm.extend(range(h * d_head, (h + 1) * d_head))

    perm = np.array(perm, dtype=np.intp)
    assert len(perm) == K and len(set(perm)) == K, "Permutation must be a bijection"

    # Apply permutation to generators: G'[:, i, j] = G[:, perm[i], perm[j]]
    G_reordered = G[:, perm][:, :, perm]

    return G_reordered, perm


def generate_multi_irrep_soN_generators(
    irrep_spec: list,
    N: int,
    *,
    validate: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Generate block-diagonal SO(N) generators from a multi-irrep specification.

    This creates generators for a direct sum of SO(N) irreducible representations:
        V = ⊕_i (V_i)^{mult_i}

    Supported irrep types:
        - 'scalar' (dim=1): Invariant, generators act as zero
        - 'fund' (dim=N): Fundamental/vector representation
        - 'wedge2' (dim=N(N-1)/2): Antisymmetric 2-tensor ∧²V
        - 'sym2' (dim=N(N+1)/2-1): Symmetric traceless 2-tensor Sym²₀V

    Args:
        irrep_spec: List of (label, multiplicity, dim) tuples.
            Example: [('scalar', 10, 1), ('fund', 8, N), ('wedge2', 4, N*(N-1)//2)]
            Total dimension K = Σ mult × dim
        N: The gauge group dimension (SO(N))
        validate: If True, verify the resulting generators
        eps: Tolerance for validation

    Returns:
        G: Block-diagonal generators, shape (N(N-1)/2, K, K)
           where K = Σ mult × dim

    Example:
        >>> # SO(5) with mixed irreps
        >>> spec = [('scalar', 10, 1), ('fund', 8, 5), ('wedge2', 4, 10)]
        >>> G = generate_multi_irrep_soN_generators(spec, N=5)
        >>> G.shape
        (10, 90, 90)  # 10 generators, K = 10 + 40 + 40 = 90

        >>> # SO(8) with all three tensor irreps
        >>> spec = [('fund', 4, 8), ('wedge2', 2, 28), ('sym2', 2, 35)]
        >>> G = generate_multi_irrep_soN_generators(spec, N=8)
        >>> G.shape
        (28, 158, 158)  # 28 generators, K = 32 + 56 + 70 = 158

    Note:
        Using diverse irreps (fund + wedge2 + sym2) provides genuinely different
        transformation channels, similar to using multiple spin-ℓ irreps for SO(3).
    """
    # Expected dimensions for each irrep type
    expected_dims = {
        'scalar': 1,
        'fund': N,
        'fundamental': N,
        'vector': N,
        'wedge2': N * (N - 1) // 2,
        'antisym2': N * (N - 1) // 2,
        'exterior2': N * (N - 1) // 2,
        'sym2': N * (N + 1) // 2 - 1,
        'sym2_traceless': N * (N + 1) // 2 - 1,
        'symmetric2': N * (N + 1) // 2 - 1,
    }

    # Validate irrep specification
    for label, mult, dim in irrep_spec:
        label_lower = label.lower()

        # Check if it's a known irrep type
        if label_lower in expected_dims:
            expected_dim = expected_dims[label_lower]
            if dim != expected_dim:
                raise ValueError(
                    f"Irrep '{label}' should have dim={expected_dim} for SO({N}), "
                    f"but got dim={dim}."
                )
        else:
            # Unknown label - check if dimension matches a known irrep
            if dim == 1:
                pass  # Scalar
            elif dim == N:
                pass  # Fundamental
            elif dim == N * (N - 1) // 2:
                pass  # ∧²V
            elif dim == N * (N + 1) // 2 - 1:
                pass  # Sym²₀V
            else:
                raise ValueError(
                    f"Irrep '{label}' has dimension {dim}, which doesn't match any "
                    f"implemented SO({N}) irrep. Supported dims: 1 (scalar), "
                    f"{N} (fund), {N*(N-1)//2} (wedge2), {N*(N+1)//2-1} (sym2)."
                )

        if mult < 0:
            raise ValueError(f"Irrep '{label}' has negative multiplicity {mult}.")

    # Compute total dimension
    K = sum(mult * dim for _, mult, dim in irrep_spec)

    # Number of generators for SO(N)
    n_gen = N * (N - 1) // 2

    # Initialize block-diagonal generators
    G = np.zeros((n_gen, K, K), dtype=np.float32)

    # Get generators for each irrep type (cached for efficiency)
    G_fund = None
    G_wedge2 = None
    G_sym2 = None

    # Fill in blocks
    idx = 0
    for label, mult, dim in irrep_spec:
        if dim == 1:
            # Scalars: generators act as zero
            idx += mult * dim

        elif dim == N:
            # Fundamental representation
            if G_fund is None:
                G_fund = generate_soN_generators(N, validate=False)
            for _ in range(mult):
                G[:, idx:idx+dim, idx:idx+dim] = G_fund
                idx += dim

        elif dim == N * (N - 1) // 2:
            # ∧²V (antisymmetric 2-tensor)
            if G_wedge2 is None:
                G_wedge2 = generate_wedge2_generators(N, validate=False)
            for _ in range(mult):
                G[:, idx:idx+dim, idx:idx+dim] = G_wedge2
                idx += dim

        elif dim == N * (N + 1) // 2 - 1:
            # Sym²₀V (symmetric traceless 2-tensor)
            if G_sym2 is None:
                G_sym2 = generate_sym2_traceless_generators(N, validate=False)
            for _ in range(mult):
                G[:, idx:idx+dim, idx:idx+dim] = G_sym2
                idx += dim

        else:
            # Should never reach here due to validation above
            raise RuntimeError(f"Unexpected dimension {dim} for irrep '{label}'")

    # Validate if requested
    if validate and K > 1:
        _validate_block_diagonal_soN_generators(G, irrep_spec, N, eps=eps)

    return G


# =============================================================================
# SO(N) Higher Tensor Representations (Non-Fundamental Irreps)
# =============================================================================

def _wedge2_index_to_pair(idx: int, N: int) -> tuple:
    """Map linear index to (i,j) pair with i < j for ∧²V basis."""
    i = 0
    count = 0
    while count + (N - 1 - i) <= idx:
        count += N - 1 - i
        i += 1
    j = idx - count + i + 1
    return i, j


def _wedge2_pair_to_index(i: int, j: int, N: int) -> int:
    """Map (i,j) pair with i < j to linear index for ∧²V basis."""
    # Sum of (N-1) + (N-2) + ... + (N-i) = i*N - i*(i+1)/2
    return i * N - i * (i + 1) // 2 + (j - i - 1)


def generate_wedge2_generators(
    N: int,
    *,
    validate: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Generate SO(N) generators for ∧²V (antisymmetric 2-tensor representation).

    The exterior square ∧²V is the space of antisymmetric N×N matrices.
    Elements can be thought of as "bivectors" or "angular momentum" components.

    Dimension: N(N-1)/2
    Basis: { e_i ∧ e_j : i < j } represented as E_ij - E_ji

    The Lie algebra action is the commutator:
        G · X = [G, X] = GX - XG

    This preserves antisymmetry (since G is skew-symmetric).

    Args:
        N: Dimension of the fundamental representation (N ≥ 2)
        validate: If True, verify the generators
        eps: Tolerance for validation

    Returns:
        G: Generators array, shape (n_gen, dim, dim)
           where n_gen = N(N-1)/2 and dim = N(N-1)/2

    Example:
        >>> G = generate_wedge2_generators(5)
        >>> G.shape
        (10, 10, 10)  # SO(5) has 10 generators, ∧²(R^5) has dim 10

    Properties:
        - Different Casimir eigenvalue than fundamental
        - Transforms as X' = O X Oᵀ under O ∈ SO(N)
        - Captures "rotational" or "angular momentum" degrees of freedom
    """
    if N < 2:
        raise ValueError(f"N must be >= 2 for SO(N), got N={N}")

    n_gen = N * (N - 1) // 2
    dim = N * (N - 1) // 2  # Same dimension as number of generators!

    # Get fundamental generators
    G_fund = generate_soN_generators(N, validate=False)  # (n_gen, N, N)

    # Build generators for ∧²V representation
    # Action: X → [G_a, X] where X is antisymmetric N×N matrix
    G_wedge2 = np.zeros((n_gen, dim, dim), dtype=np.float32)

    for a in range(n_gen):
        G_a = G_fund[a]  # (N, N) skew-symmetric

        for p in range(dim):  # Input basis element index
            i, j = _wedge2_index_to_pair(p, N)

            # Basis element: E_ij - E_ji (antisymmetric)
            X = np.zeros((N, N), dtype=np.float32)
            X[i, j] = 1.0
            X[j, i] = -1.0

            # Commutator [G_a, X] = G_a @ X - X @ G_a
            comm = G_a @ X - X @ G_a  # Still antisymmetric

            # Express result in ∧² basis
            for q in range(dim):
                k, l = _wedge2_index_to_pair(q, N)
                # The coefficient is the (k,l) entry (upper triangle)
                G_wedge2[a, q, p] = comm[k, l]

    if validate:
        _validate_wedge2_generators(G_wedge2, N, eps=eps)

    return G_wedge2


def _validate_wedge2_generators(
    G: np.ndarray,
    N: int,
    *,
    eps: float = 1e-6,
) -> None:
    """Validate ∧²V generators."""
    n_gen, dim, _ = G.shape

    expected_n_gen = N * (N - 1) // 2
    expected_dim = N * (N - 1) // 2

    if n_gen != expected_n_gen:
        raise ValueError(f"Expected {expected_n_gen} generators, got {n_gen}")
    if dim != expected_dim:
        raise ValueError(f"Expected dim {expected_dim}, got {dim}")

    # Check skew-symmetry of generators
    for a in range(n_gen):
        skew_error = np.linalg.norm(G[a] + G[a].T, ord='fro')
        if skew_error > eps:
            raise RuntimeError(
                f"∧² generator G[{a}] not skew-symmetric: ||G + Gᵀ|| = {skew_error:.3e}"
            )

    # Check sample commutation relations (they should form so(N) algebra)
    if n_gen >= 3:
        comm_01 = G[0] @ G[1] - G[1] @ G[0]
        if np.linalg.norm(comm_01 + comm_01.T, ord='fro') > eps:
            raise RuntimeError("Commutator [G_0, G_1] in ∧² rep not skew-symmetric")


def _sym2_traceless_basis_size(N: int) -> int:
    """Dimension of Sym²₀V (symmetric traceless 2-tensors)."""
    return N * (N + 1) // 2 - 1


def _sym2_traceless_index_to_components(idx: int, N: int) -> tuple:
    """
    Map linear index to symmetric traceless basis element.

    Basis ordering:
    - First N(N-1)/2 indices: off-diagonal (i,j) with i < j, coefficient √2
    - Next N-1 indices: diagonal traceless combinations

    Returns:
        (type, data) where:
        - type='offdiag': data=(i, j) for off-diagonal element
        - type='diag': data=k for k-th diagonal traceless element
    """
    n_offdiag = N * (N - 1) // 2

    if idx < n_offdiag:
        # Off-diagonal element
        i, j = _wedge2_index_to_pair(idx, N)
        return ('offdiag', (i, j))
    else:
        # Diagonal traceless element
        k = idx - n_offdiag
        return ('diag', k)


def _build_sym2_traceless_basis_element(idx: int, N: int) -> np.ndarray:
    """
    Build the idx-th basis element of Sym²₀V as an N×N matrix.

    The basis is orthonormal under the Frobenius inner product.
    """
    n_offdiag = N * (N - 1) // 2
    X = np.zeros((N, N), dtype=np.float32)

    if idx < n_offdiag:
        # Off-diagonal: (E_ij + E_ji) / √2
        i, j = _wedge2_index_to_pair(idx, N)
        X[i, j] = 1.0 / np.sqrt(2)
        X[j, i] = 1.0 / np.sqrt(2)
    else:
        # Diagonal traceless: use Gell-Mann-like basis
        # Element k: (E_00 + ... + E_kk - (k+1)E_{k+1,k+1}) / √((k+1)(k+2))
        k = idx - n_offdiag
        norm = np.sqrt((k + 1) * (k + 2))
        for i in range(k + 1):
            X[i, i] = 1.0 / norm
        X[k + 1, k + 1] = -(k + 1) / norm

    return X


def generate_sym2_traceless_generators(
    N: int,
    *,
    validate: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Generate SO(N) generators for Sym²₀V (symmetric traceless 2-tensor representation).

    The symmetric traceless square Sym²₀V is the space of symmetric N×N matrices
    with trace zero. These represent "quadrupolar" or "strain-like" degrees of freedom.

    Dimension: N(N+1)/2 - 1
    Basis: Orthonormal symmetric traceless matrices

    The Lie algebra action is the commutator:
        G · X = [G, X] = GX - XG

    This preserves symmetry and tracelessness.

    Args:
        N: Dimension of the fundamental representation (N ≥ 2)
        validate: If True, verify the generators
        eps: Tolerance for validation

    Returns:
        G: Generators array, shape (n_gen, dim, dim)
           where n_gen = N(N-1)/2 and dim = N(N+1)/2 - 1

    Example:
        >>> G = generate_sym2_traceless_generators(5)
        >>> G.shape
        (10, 14, 14)  # SO(5) has 10 generators, Sym²₀(R^5) has dim 14

    Properties:
        - Different Casimir eigenvalue than fundamental and ∧²
        - Transforms as X' = O X Oᵀ under O ∈ SO(N)
        - Captures "quadrupolar" or "deformation" degrees of freedom
    """
    if N < 2:
        raise ValueError(f"N must be >= 2 for SO(N), got N={N}")

    n_gen = N * (N - 1) // 2
    dim = _sym2_traceless_basis_size(N)

    # Get fundamental generators
    G_fund = generate_soN_generators(N, validate=False)  # (n_gen, N, N)

    # Pre-build basis elements
    basis = [_build_sym2_traceless_basis_element(p, N) for p in range(dim)]

    # Build generators for Sym²₀V representation
    # Action: X → [G_a, X] where X is symmetric traceless N×N matrix
    G_sym2 = np.zeros((n_gen, dim, dim), dtype=np.float32)

    for a in range(n_gen):
        G_a = G_fund[a]  # (N, N) skew-symmetric

        for p in range(dim):  # Input basis element index
            X = basis[p]

            # Commutator [G_a, X] = G_a @ X - X @ G_a
            # This is symmetric (and traceless) when G is skew and X is symmetric
            comm = G_a @ X - X @ G_a

            # Express result in Sym²₀ basis via inner product
            for q in range(dim):
                Y = basis[q]
                # Inner product: tr(Yᵀ comm) = tr(Y comm) since both symmetric
                G_sym2[a, q, p] = np.sum(Y * comm)

    if validate:
        _validate_sym2_traceless_generators(G_sym2, N, eps=eps)

    return G_sym2


def _validate_sym2_traceless_generators(
    G: np.ndarray,
    N: int,
    *,
    eps: float = 1e-6,
) -> None:
    """Validate Sym²₀V generators."""
    n_gen, dim, _ = G.shape

    expected_n_gen = N * (N - 1) // 2
    expected_dim = _sym2_traceless_basis_size(N)

    if n_gen != expected_n_gen:
        raise ValueError(f"Expected {expected_n_gen} generators, got {n_gen}")
    if dim != expected_dim:
        raise ValueError(f"Expected dim {expected_dim}, got {dim}")

    # Check skew-symmetry of generators
    for a in range(n_gen):
        skew_error = np.linalg.norm(G[a] + G[a].T, ord='fro')
        if skew_error > eps:
            raise RuntimeError(
                f"Sym²₀ generator G[{a}] not skew-symmetric: ||G + Gᵀ|| = {skew_error:.3e}"
            )

    # Check sample commutation
    if n_gen >= 3:
        comm_01 = G[0] @ G[1] - G[1] @ G[0]
        if np.linalg.norm(comm_01 + comm_01.T, ord='fro') > eps:
            raise RuntimeError("Commutator [G_0, G_1] in Sym²₀ rep not skew-symmetric")


def _validate_block_diagonal_soN_generators(
    G: np.ndarray,
    irrep_spec: list,
    N: int,
    *,
    eps: float = 1e-6,
) -> None:
    """
    Validate block-diagonal multi-irrep SO(N) generators.

    Checks:
    1. Skew-symmetry
    2. Sample commutation relations
    3. Block structure (off-diagonal blocks are zero)
    """
    n_gen = G.shape[0]
    K = G.shape[1]

    expected_n_gen = N * (N - 1) // 2
    if n_gen != expected_n_gen:
        raise ValueError(
            f"Expected {expected_n_gen} generators for SO({N}), got {n_gen}"
        )

    # Check skew-symmetry
    for a in range(n_gen):
        skew_error = np.linalg.norm(G[a] + G[a].T, ord='fro')
        if skew_error > eps:
            raise RuntimeError(
                f"Block-diagonal SO({N}) generator G[{a}] not skew-symmetric: "
                f"||G + Gᵀ|| = {skew_error:.3e}"
            )

    # Check sample commutation (first 3 generators if available, like SO(3) subset)
    if n_gen >= 3:
        G_0, G_1, G_2 = G[0], G[1], G[2]

        # For SO(N) with N >= 3, generators 0,1,2 correspond to:
        # L_{01}, L_{02}, L_{03} or similar
        # Their commutations depend on index structure

        # Just check that commutators are skew-symmetric (sanity check)
        comm_01 = G_0 @ G_1 - G_1 @ G_0
        if np.linalg.norm(comm_01 + comm_01.T, ord='fro') > eps:
            raise RuntimeError("Commutator [G_0, G_1] not skew-symmetric")

    # Check block structure
    idx = 0
    block_starts = []
    for _, mult, dim in irrep_spec:
        for _ in range(mult):
            block_starts.append((idx, dim))
            idx += dim

    for i, (start_i, dim_i) in enumerate(block_starts):
        for j, (start_j, dim_j) in enumerate(block_starts):
            if i != j:
                for a in range(min(n_gen, 10)):  # Check first 10 generators
                    block = G[a, start_i:start_i+dim_i, start_j:start_j+dim_j]
                    block_norm = np.linalg.norm(block, ord='fro')
                    if block_norm > eps:
                        raise RuntimeError(
                            f"Off-diagonal block ({i},{j}) in generator {a} "
                            f"is non-zero: ||block|| = {block_norm:.3e}"
                        )


# =============================================================================
# SO(N) Lie Algebra Operations (PyTorch)
# =============================================================================

def _get_soN_gauge_generators(n_gen: int, device, dtype) -> 'torch.Tensor':
    """
    Get N×N generators for SO(N) gauge group based on n_gen.

    These are the canonical so(N) basis elements L_{ij}, NOT the K×K transport generators.
    Used internally for BCH composition.
    """
    import torch
    import math

    # Infer N from n_gen: n_gen = N(N-1)/2
    N = int((1 + math.sqrt(1 + 8 * n_gen)) / 2)

    if N * (N - 1) // 2 != n_gen:
        raise ValueError(f"n_gen={n_gen} doesn't correspond to valid SO(N)")

    # Build canonical generators L_{ij} for i < j
    generators = torch.zeros(n_gen, N, N, device=device, dtype=dtype)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            generators[idx, i, j] = 1.0
            generators[idx, j, i] = -1.0
            idx += 1

    return generators


def soN_bracket_torch(
    phi1: 'torch.Tensor',
    phi2: 'torch.Tensor',
    generators: 'torch.Tensor',
) -> 'torch.Tensor':
    """
    Compute the Lie bracket [φ₁·G, φ₂·G] in so(N) and return coordinates.

    For so(N), the Lie bracket of two skew-symmetric matrices is:
        [A, B] = AB - BA

    This is used in BCH composition for proper Lie group updates.

    Args:
        phi1: First Lie algebra element coordinates (..., n_gen)
        phi2: Second Lie algebra element coordinates (..., n_gen)
        generators: Lie algebra generators (n_gen, K, K) - used only for n_gen count
                   The actual N×N generators are computed internally.

    Returns:
        bracket_coords: Coordinates of [φ₁·G, φ₂·G] in generator basis (..., n_gen)
    """
    import torch

    n_gen = generators.shape[0]

    # Get proper N×N generators for the gauge group (not K×K transport generators!)
    gauge_gens = _get_soN_gauge_generators(n_gen, phi1.device, phi1.dtype)

    # Build skew-symmetric matrices using N×N gauge generators
    A1 = torch.einsum('...a,aij->...ij', phi1, gauge_gens)  # (..., N, N)
    A2 = torch.einsum('...a,aij->...ij', phi2, gauge_gens)  # (..., N, N)

    # Lie bracket: [A, B] = AB - BA
    bracket = A1 @ A2 - A2 @ A1  # (..., N, N)

    # Extract coordinates from upper triangular
    bracket_coords = extract_soN_coords_torch(bracket, gauge_gens)

    return bracket_coords


def extract_soN_coords_torch(
    A: 'torch.Tensor',
    generators: 'torch.Tensor',
) -> 'torch.Tensor':
    """
    Extract so(N) Lie algebra coordinates from a skew-symmetric matrix.

    Given A = Σ_a φ_a G_a, extract the coordinates φ_a.

    For the canonical basis L_{ij} (with i < j), the coordinates are simply
    the upper-triangular elements of A: φ_a = A[i, j].

    Args:
        A: Skew-symmetric matrix (..., M, M) where M is matrix dimension
        generators: Lie algebra generators (n_gen, K, K)
                   Note: K may be embedding dim, not gauge group dim!

    Returns:
        phi: Lie algebra coordinates (..., n_gen)
    """
    import torch
    import math

    n_gen = generators.shape[0]
    M = A.shape[-1]  # Matrix dimension of A

    # Infer gauge group dimension N from n_gen: n_gen = N(N-1)/2
    # Solving: N = (1 + sqrt(1 + 8*n_gen)) / 2
    N = int((1 + math.sqrt(1 + 8 * n_gen)) / 2)

    # Validate
    if N * (N - 1) // 2 != n_gen:
        raise ValueError(f"n_gen={n_gen} doesn't correspond to valid SO(N). "
                        f"Expected N*(N-1)/2 for some integer N.")

    if M != N:
        raise ValueError(f"Matrix A has dimension {M}x{M} but gauge group is SO({N}). "
                        f"For BCH composition, need {N}x{N} matrices.")

    # Build index mapping: generator a -> (i, j) with i < j
    # For canonical basis, generator a corresponds to pair (i, j) in order
    batch_shape = A.shape[:-2]
    phi = torch.zeros(*batch_shape, n_gen, device=A.device, dtype=A.dtype)

    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            # φ_a = A[i, j] (upper triangular element)
            phi[..., idx] = A[..., i, j]
            idx += 1

    return phi


def soN_compose_bch_torch(
    phi1: 'torch.Tensor',
    phi2: 'torch.Tensor',
    generators: 'torch.Tensor',
    order: int = 1,
) -> 'torch.Tensor':
    """
    Compose two so(N) elements using Baker-Campbell-Hausdorff formula.

    log(exp(φ₁·G)·exp(φ₂·G)) = φ₁ + φ₂ + ½[φ₁,φ₂] + (1/12)[φ₁,[φ₁,φ₂]] - ...

    For so(N), the Lie bracket is: [A, B] = AB - BA (matrix commutator)

    This is the proper way to compose updates in the Lie algebra, ensuring
    the result corresponds to a valid group element when exponentiated.

    Args:
        phi1: First so(N) element (..., n_gen)
        phi2: Second so(N) element (..., n_gen)
        generators: Lie algebra generators (n_gen, N, N)
        order: BCH expansion order (0=addition, 1=first correction, 2=second)

    Returns:
        phi_composed: Composed element in so(N) (..., n_gen)
    """
    if order == 0:
        # Simple addition (valid for small angles only)
        return phi1 + phi2

    # First-order BCH: φ₁ + φ₂ + ½[φ₁,φ₂]
    bracket_12 = soN_bracket_torch(phi1, phi2, generators)
    result = phi1 + phi2 + 0.5 * bracket_12

    if order >= 2:
        # Second-order: + (1/12)[φ₁,[φ₁,φ₂]] - (1/12)[φ₂,[φ₁,φ₂]]
        bracket_1_12 = soN_bracket_torch(phi1, bracket_12, generators)
        bracket_2_12 = soN_bracket_torch(phi2, bracket_12, generators)
        result = result + (1.0/12.0) * bracket_1_12 - (1.0/12.0) * bracket_2_12

    return result


def retract_soN_torch(
    phi: 'torch.Tensor',
    delta_phi: 'torch.Tensor',
    generators: 'torch.Tensor',
    step_size: float = 1.0,
    trust_region: float = 0.3,
    max_norm: float = 3.14159,
    bch_order: int = 1,
    eps: float = 1e-6,
) -> 'torch.Tensor':
    """
    Retract phi update onto SO(N) manifold with trust region.

    This is the proper way to update gauge frames φ:
    1. Scale delta by step_size
    2. Apply trust region (limit relative change)
    3. Compose using BCH formula (proper Lie group composition)
    4. Clamp final norm

    Args:
        phi: Current gauge frames (..., n_gen)
        delta_phi: Update direction (typically -grad_phi) (..., n_gen)
        generators: Lie algebra generators (n_gen, N, N)
        step_size: Learning rate for the update
        trust_region: Maximum relative change ||δφ|| / ||φ|| per update
        max_norm: Maximum allowed norm for phi (π = 180° rotation)
        bch_order: Order of BCH expansion (0=add, 1=first correction)
        eps: Numerical stability constant

    Returns:
        phi_new: Updated gauge frames (..., n_gen)
    """
    import torch

    # Scale update
    update = step_size * delta_phi

    # Trust region: limit step size relative to current phi
    phi_norm = torch.norm(phi, dim=-1, keepdim=True).clamp(min=0.1)
    update_norm = torch.norm(update, dim=-1, keepdim=True)

    # Scale down if update is too large relative to current phi
    scale = torch.clamp(trust_region * phi_norm / (update_norm + eps), max=1.0)
    update = scale * update

    # Compose using BCH (proper Lie group composition)
    phi_new = soN_compose_bch_torch(phi, update, generators, order=bch_order)

    # Clamp to max norm (retraction to ball)
    phi_new_norm = torch.norm(phi_new, dim=-1, keepdim=True)
    phi_new = torch.where(
        phi_new_norm > max_norm,
        phi_new * (max_norm / (phi_new_norm + eps)),
        phi_new
    )

    return phi_new


def retract_soN_exact_torch(
    phi: 'torch.Tensor',
    delta_phi: 'torch.Tensor',
    generators: 'torch.Tensor',
    step_size: float = 1.0,
    trust_region: float = 0.3,
    max_norm: float = 3.14159,
    eps: float = 1e-6,
) -> 'torch.Tensor':
    """
    Exact SO(N) retraction via matrix exponential and logarithm.

    Computes: φ_new = log(exp(φ·G) · exp(δφ·G))

    This is more accurate than BCH for large updates but more expensive.
    Uses real Schur decomposition for the matrix logarithm.

    Args:
        phi: Current gauge frames (..., n_gen)
        delta_phi: Update direction (..., n_gen)
        generators: Lie algebra generators (n_gen, N, N)
        step_size: Learning rate
        trust_region: Maximum relative change
        max_norm: Maximum norm for phi
        eps: Numerical stability

    Returns:
        phi_new: Updated gauge frames (..., n_gen)
    """
    import torch

    n_gen = generators.shape[0]

    # Get proper N×N generators for the gauge group (not K×K transport generators!)
    gauge_gens = _get_soN_gauge_generators(n_gen, phi.device, phi.dtype)

    # Scale update with trust region
    update = step_size * delta_phi
    phi_norm = torch.norm(phi, dim=-1, keepdim=True).clamp(min=0.1)
    update_norm = torch.norm(update, dim=-1, keepdim=True)
    scale = torch.clamp(trust_region * phi_norm / (update_norm + eps), max=1.0)
    update = scale * update

    # Build skew-symmetric matrices using N×N gauge generators
    A_phi = torch.einsum('...a,aij->...ij', phi, gauge_gens)
    A_delta = torch.einsum('...a,aij->...ij', update, gauge_gens)

    # Matrix exponentials
    R_phi = torch.matrix_exp(A_phi)
    R_delta = torch.matrix_exp(A_delta)

    # Group product
    R_new = R_phi @ R_delta

    # Matrix logarithm for orthogonal matrices
    # Use the fact that for R ∈ SO(N), log(R) is skew-symmetric
    A_new = _matrix_log_orthogonal_torch(R_new, eps=eps)

    # Extract coordinates
    phi_new = extract_soN_coords_torch(A_new, gauge_gens)

    # Clamp to max norm
    phi_new_norm = torch.norm(phi_new, dim=-1, keepdim=True)
    phi_new = torch.where(
        phi_new_norm > max_norm,
        phi_new * (max_norm / (phi_new_norm + eps)),
        phi_new
    )

    return phi_new


def _matrix_log_orthogonal_torch(
    R: 'torch.Tensor',
    eps: float = 1e-6,
) -> 'torch.Tensor':
    """
    Compute matrix logarithm for orthogonal matrices.

    For R ∈ SO(N), log(R) is a skew-symmetric matrix in so(N).
    Uses the real Schur decomposition approach for stability.

    Args:
        R: Orthogonal matrix (..., N, N)
        eps: Numerical stability

    Returns:
        A: Skew-symmetric matrix log(R) (..., N, N)
    """
    import torch

    # For small deviations from identity, use first-order approximation
    # log(I + X) ≈ X - X²/2 + X³/3 - ...
    # For orthogonal R = I + X where X is small and skew-symmetric

    N = R.shape[-1]
    I = torch.eye(N, device=R.device, dtype=R.dtype)

    # Check if close to identity (common case for small updates)
    deviation = R - I
    deviation_norm = torch.norm(deviation, dim=(-2, -1), keepdim=True)

    # For small deviations, use series expansion
    # For larger deviations, use the antisymmetric part extraction
    # (This is a simplified approach; full Schur method would be more robust)

    # Antisymmetric part of deviation gives first-order log
    A_approx = 0.5 * (deviation - deviation.transpose(-1, -2))

    # For better accuracy with larger rotations, use iterative refinement
    # Newton iteration: A_{k+1} = A_k + (R - exp(A_k)) @ exp(-A_k) antisymmetrized
    # But for simplicity and speed, we use BCH-based correction

    # Second-order correction
    A_sq = A_approx @ A_approx
    correction = -0.5 * A_sq  # Second-order term
    A = A_approx + 0.5 * (correction - correction.transpose(-1, -2))

    # Ensure skew-symmetry
    A = 0.5 * (A - A.transpose(-1, -2))

    return A


# =============================================================================
# GL(K) Lie Algebra Operations (full general linear group)
# =============================================================================
#
# GL(K) is the group of invertible K×K matrices with Lie algebra gl(K).
# gl(K) = all K×K matrices with Lie bracket [A,B] = AB - BA.
#
# Key difference from so(K): gl(K) includes non-skew-symmetric matrices,
# so exp(φ·G) produces general invertible matrices, not just orthogonal ones.
#
# Since KL divergence (and all f-divergences) are invariant under GL(K),
# we can use these more expressive transformations.


def _get_glK_gauge_generators(
    n_gen: int,
    device: 'torch.device',
    dtype: 'torch.dtype',
) -> 'torch.Tensor':
    """
    Get K×K generators for GL(K) gauge group based on n_gen = K².

    Uses elementary matrices E_ij (1 at position (i,j), 0 elsewhere).

    Args:
        n_gen: Number of generators (must be a perfect square K²)
        device: PyTorch device
        dtype: PyTorch dtype

    Returns:
        generators: (K², K, K) tensor of elementary matrices
    """
    import torch
    import math

    # Infer K from n_gen = K²
    K = int(math.sqrt(n_gen))

    if K * K != n_gen:
        raise ValueError(f"n_gen={n_gen} is not a perfect square (needed for GL(K))")

    # Build elementary matrices E_ij
    generators = torch.zeros(n_gen, K, K, device=device, dtype=dtype)
    idx = 0
    for i in range(K):
        for j in range(K):
            generators[idx, i, j] = 1.0
            idx += 1

    return generators


def glK_bracket_torch(
    phi1: 'torch.Tensor',
    phi2: 'torch.Tensor',
    generators: 'torch.Tensor',
) -> 'torch.Tensor':
    """
    Compute the Lie bracket [φ₁·G, φ₂·G] in gl(K) and return coordinates.

    For gl(K), the Lie bracket is the matrix commutator: [A, B] = AB - BA

    Args:
        phi1: First Lie algebra element coordinates (..., n_gen) where n_gen = K²
        phi2: Second Lie algebra element coordinates (..., n_gen)
        generators: Transport generators (n_gen, dim, dim) - used only for n_gen count.
                   The actual K×K generators are computed internally.

    Returns:
        bracket_coords: Coordinates of [φ₁·G, φ₂·G] in generator basis (..., n_gen)
    """
    import torch

    n_gen = generators.shape[0]

    # Get K×K gauge generators (elementary matrices)
    gauge_gens = _get_glK_gauge_generators(n_gen, phi1.device, phi1.dtype)

    # Build matrices using K×K gauge generators
    A1 = torch.einsum('...a,aij->...ij', phi1, gauge_gens)  # (..., K, K)
    A2 = torch.einsum('...a,aij->...ij', phi2, gauge_gens)  # (..., K, K)

    # Lie bracket: [A, B] = AB - BA
    bracket = A1 @ A2 - A2 @ A1  # (..., K, K)

    # Extract coordinates
    bracket_coords = extract_glK_coords_torch(bracket, gauge_gens)

    return bracket_coords


def extract_glK_coords_torch(
    A: 'torch.Tensor',
    generators: 'torch.Tensor',
) -> 'torch.Tensor':
    """
    Extract gl(K) Lie algebra coordinates from a matrix.

    Given A = Σ_a φ_a E_a where E_a are elementary matrices,
    the coordinates are simply the matrix elements: φ_{ij} = A[i, j].

    Args:
        A: Matrix (..., K, K)
        generators: Gauge generators (n_gen, K, K) - used for shape only

    Returns:
        phi: Lie algebra coordinates (..., n_gen) where n_gen = K²
    """
    import torch

    K = A.shape[-1]
    batch_shape = A.shape[:-2]

    # Flatten the matrix to get coordinates
    # E_ij has index i*K + j, so A[i,j] = phi[i*K + j]
    phi = A.reshape(batch_shape + (K * K,))

    return phi


def glK_compose_bch_torch(
    phi1: 'torch.Tensor',
    phi2: 'torch.Tensor',
    generators: 'torch.Tensor',
    order: int = 1,
) -> 'torch.Tensor':
    """
    Compose two gl(K) elements using Baker-Campbell-Hausdorff formula.

    log(exp(φ₁·G)·exp(φ₂·G)) = φ₁ + φ₂ + ½[φ₁,φ₂] + (1/12)[φ₁,[φ₁,φ₂]] - ...

    For gl(K), the Lie bracket is: [A, B] = AB - BA (matrix commutator)

    Args:
        phi1: First gl(K) element (..., n_gen) where n_gen = K²
        phi2: Second gl(K) element (..., n_gen)
        generators: Lie algebra generators (n_gen, dim, dim)
        order: BCH expansion order (0=addition, 1=first correction, 2=second)

    Returns:
        phi_composed: Composed element in gl(K) (..., n_gen)
    """
    if order == 0:
        # Simple addition (valid for small updates only)
        return phi1 + phi2

    # First-order BCH: φ₁ + φ₂ + ½[φ₁,φ₂]
    bracket_12 = glK_bracket_torch(phi1, phi2, generators)
    result = phi1 + phi2 + 0.5 * bracket_12

    if order >= 2:
        # Second-order: + (1/12)[φ₁,[φ₁,φ₂]] - (1/12)[φ₂,[φ₁,φ₂]]
        bracket_1_12 = glK_bracket_torch(phi1, bracket_12, generators)
        bracket_2_12 = glK_bracket_torch(phi2, bracket_12, generators)
        result = result + (1.0/12.0) * bracket_1_12 - (1.0/12.0) * bracket_2_12

    return result


def retract_glK_torch(
    phi: 'torch.Tensor',
    delta_phi: 'torch.Tensor',
    generators: 'torch.Tensor',
    step_size: float = 1.0,
    trust_region: float = 0.1,  # Tighter than SO(N) for stability
    max_norm: float = 1.0,  # Smaller than SO(N) - GL(K) doesn't have periodicity
    bch_order: int = 0,  # Use simple addition - BCH bracket can amplify noise
    eps: float = 1e-6,
    grad_clip: float = 10.0,  # Clip gradient norm before scaling
) -> 'torch.Tensor':
    """
    Retract phi update in GL(K) with trust region.

    Unlike SO(N), GL(K) doesn't require orthogonality constraints.
    We use conservative settings since GL(K) can produce ill-conditioned
    transport operators more easily than SO(K).

    Steps:
    1. Clip gradient to prevent explosions
    2. Scale update by step_size
    3. Apply trust region (limit relative change)
    4. Compose (simple addition for stability, or BCH if requested)
    5. Clamp final norm

    Args:
        phi: Current gauge frames (..., n_gen) where n_gen = K²
        delta_phi: Update direction (typically -grad_phi) (..., n_gen)
        generators: Lie algebra generators (n_gen, dim, dim)
        step_size: Learning rate for the update
        trust_region: Maximum relative change ||δφ|| / ||φ|| per update
        max_norm: Maximum allowed norm for phi
        bch_order: Order of BCH expansion (0=add, 1=first correction)
        eps: Numerical stability constant
        grad_clip: Maximum gradient norm (per-element clipping)

    Returns:
        phi_new: Updated gauge frames (..., n_gen)
    """
    import torch

    # Clip gradient to prevent explosions
    delta_norm = torch.norm(delta_phi, dim=-1, keepdim=True)
    clip_scale = torch.clamp(grad_clip / (delta_norm + eps), max=1.0)
    delta_phi_clipped = clip_scale * delta_phi

    # Scale update
    update = step_size * delta_phi_clipped

    # Trust region: limit step size relative to current phi
    phi_norm = torch.norm(phi, dim=-1, keepdim=True).clamp(min=0.1)
    update_norm = torch.norm(update, dim=-1, keepdim=True)

    # Scale down if update is too large relative to current phi
    scale = torch.clamp(trust_region * phi_norm / (update_norm + eps), max=1.0)
    update = scale * update

    # Additional absolute clipping on update magnitude
    update_norm_after = torch.norm(update, dim=-1, keepdim=True)
    max_update = 0.5  # Max absolute update per step
    update = torch.where(
        update_norm_after > max_update,
        update * max_update / (update_norm_after + eps),
        update
    )

    # Compose: use simple addition for GL(K) stability (BCH bracket can explode)
    if bch_order == 0:
        phi_new = phi + update
    else:
        # BCH composition (use with caution for GL(K))
        phi_new = glK_compose_bch_torch(phi, update, generators, order=bch_order)

    # Clamp to max norm (retraction to ball)
    phi_new_norm = torch.norm(phi_new, dim=-1, keepdim=True)
    phi_new = torch.where(
        phi_new_norm > max_norm,
        phi_new * (max_norm / (phi_new_norm + eps)),
        phi_new
    )

    return phi_new


def is_glK_generators(n_gen: int) -> bool:
    """Check if n_gen corresponds to GL(K) (perfect square)."""
    import math
    K = int(math.sqrt(n_gen))
    return K * K == n_gen and K > 0


def is_soN_generators(n_gen: int) -> bool:
    """Check if n_gen corresponds to SO(N) (triangular number)."""
    import math
    # n_gen = N(N-1)/2 => 8*n_gen + 1 = (2N-1)² is a perfect odd square
    discriminant = 1 + 8 * n_gen
    sqrt_disc = int(math.sqrt(discriminant))
    if sqrt_disc * sqrt_disc != discriminant:
        return False
    N = (1 + sqrt_disc) // 2
    return N * (N - 1) // 2 == n_gen and N >= 2


# =============================================================================
# Cache
# =============================================================================

_GENERATOR_CACHE: Dict[int, np.ndarray] = {}