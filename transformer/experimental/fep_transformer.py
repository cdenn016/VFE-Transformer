"""
Pure Free Energy Principle Transformer - Built from First Principles

This implements a transformer where ALL dynamics emerge from minimizing a single
Variational Free Energy functional. No ad hoc neural network components.

Mathematical Foundation:
========================
The VFE functional over beliefs q_i = N(μ_i, Σ_i) with gauge frames φ_i:

F[q] = Σ_i F_self[q_i]                           # Self-coupling (uncertainty cost)
     + Σ_{i,j} F_align[q_i, q_j; Ω_ij]           # Belief alignment (attention)
     + Σ_i F_prior[q_i, π_i]                     # Prior coupling (memory)
     + Σ_i F_obs[q_i, y_i]                       # Observation (prediction)

where:
- Ω_ij = exp(φ_ij) is the gauge transport from frame j to frame i
- φ_ij = BCH(φ_i, -φ_j) = φ_i - φ_j - ½[φ_i, φ_j]_𝔤 + O(φ³)
- [·,·]_𝔤 is the Lie bracket with structure constants f_abc

Key Design Choices:
==================
1. SO(N) gauge group with fundamental representation
2. BCH formula for transport (not naive subtraction)
3. Block-diagonal covariance aligned with irreps
4. Haar initialization for symmetry breaking
5. Natural gradient descent respecting Fisher-Rao geometry

Author: Built from scratch following the FEP papers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass


# =============================================================================
# PART 1: SO(N) LIE ALGEBRA
# =============================================================================

def so_n_generators(n: int, device='cpu', dtype=torch.float32) -> torch.Tensor:
    """
    Generate the basis for so(n) Lie algebra (antisymmetric matrices).

    The Lie algebra so(n) has dimension n(n-1)/2.
    Basis elements T_{ab} (a < b) have:
        (T_{ab})_{ij} = δ_{ai}δ_{bj} - δ_{aj}δ_{bi}

    Args:
        n: Dimension of SO(N)

    Returns:
        generators: (n_gen, n, n) tensor where n_gen = n(n-1)/2
    """
    n_gen = n * (n - 1) // 2
    generators = torch.zeros(n_gen, n, n, device=device, dtype=dtype)

    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            generators[idx, a, b] = 1.0
            generators[idx, b, a] = -1.0
            idx += 1

    return generators


def so_n_structure_constants(n: int, device='cpu', dtype=torch.float32) -> torch.Tensor:
    """
    Compute structure constants f_abc for so(n).

    The Lie bracket is: [T_a, T_b] = Σ_c f_abc T_c

    For so(n), the structure constants come from:
        [T_{ij}, T_{kl}] = δ_{jk}T_{il} - δ_{ik}T_{jl} - δ_{jl}T_{ik} + δ_{il}T_{jk}

    Args:
        n: Dimension of SO(N)

    Returns:
        f_abc: (n_gen, n_gen, n_gen) structure constants
    """
    generators = so_n_generators(n, device, dtype)
    n_gen = generators.shape[0]

    # Compute [T_a, T_b] for all pairs
    # [A, B] = AB - BA
    commutators = torch.einsum('aij,bjk->abik', generators, generators) - \
                  torch.einsum('bij,ajk->abik', generators, generators)

    # Project onto basis to get f_abc
    # [T_a, T_b] = f_abc T_c  =>  f_abc = Tr([T_a, T_b] T_c^T) / Tr(T_c T_c^T)
    # For orthonormal basis: f_abc = -½ Tr([T_a, T_b] T_c)
    # Our generators have Tr(T_a T_b^T) = 2 δ_{ab}, so normalize

    f_abc = torch.einsum('abij,cji->abc', commutators, generators) / 2.0

    return f_abc


def lie_bracket(phi_1: torch.Tensor, phi_2: torch.Tensor,
                f_abc: torch.Tensor) -> torch.Tensor:
    """
    Compute Lie bracket [φ_1, φ_2]_𝔤 using structure constants.

    If φ_1 = α^a T_a and φ_2 = β^b T_b, then:
        [φ_1, φ_2] = α^a β^b [T_a, T_b] = α^a β^b f_abc T_c

    In coefficient form: [φ_1, φ_2]^c = f_abc α^a β^b

    Args:
        phi_1: (..., dim_g) Lie algebra coefficients
        phi_2: (..., dim_g) Lie algebra coefficients
        f_abc: (dim_g, dim_g, dim_g) structure constants

    Returns:
        bracket: (..., dim_g) coefficients of [φ_1, φ_2]
    """
    # [φ_1, φ_2]^c = f_abc φ_1^a φ_2^b
    return torch.einsum('abc,...a,...b->...c', f_abc, phi_1, phi_2)


# =============================================================================
# PART 2: BAKER-CAMPBELL-HAUSDORFF TRANSPORT
# =============================================================================

def bch_combine(phi_i: torch.Tensor, phi_j: torch.Tensor,
                f_abc: torch.Tensor, order: int = 2) -> torch.Tensor:
    """
    Combine gauge frames using Baker-Campbell-Hausdorff formula.

    BCH formula: log(exp(X)exp(Y)) = X + Y + ½[X,Y] + 1/12[X,[X,Y]] - 1/12[Y,[X,Y]] + ...

    For transport from frame j to frame i: φ_ij = BCH(φ_i, -φ_j)

    Args:
        phi_i: (..., dim_g) source frame coefficients
        phi_j: (..., dim_g) target frame coefficients
        f_abc: (dim_g, dim_g, dim_g) structure constants
        order: BCH truncation order (1=sum, 2=+commutator, 3=+nested)

    Returns:
        phi_ij: (..., dim_g) transport coefficients
    """
    # Order 1: Just sum (naive, but sometimes sufficient for small φ)
    phi_ij = phi_i - phi_j

    if order >= 2:
        # Order 2: Add first commutator term
        # BCH: X + Y + ½[X, Y] where X=φ_i, Y=-φ_j
        # = φ_i - φ_j + ½[φ_i, -φ_j] = φ_i - φ_j - ½[φ_i, φ_j]
        bracket = lie_bracket(phi_i, phi_j, f_abc)
        phi_ij = phi_ij - 0.5 * bracket

    if order >= 3:
        # Order 3: Add nested commutator terms
        # + 1/12[X,[X,Y]] - 1/12[Y,[X,Y]]
        # = 1/12[φ_i, [φ_i, -φ_j]] - 1/12[-φ_j, [φ_i, -φ_j]]
        # = -1/12[φ_i, [φ_i, φ_j]] - 1/12[φ_j, [φ_i, φ_j]]
        bracket_ij = lie_bracket(phi_i, phi_j, f_abc)
        nested_i = lie_bracket(phi_i, bracket_ij, f_abc)
        nested_j = lie_bracket(phi_j, bracket_ij, f_abc)
        phi_ij = phi_ij - (1/12) * (nested_i + nested_j)

    return phi_ij


def exp_so_n(phi: torch.Tensor, generators: torch.Tensor,
             max_terms: int = 6) -> torch.Tensor:
    """
    Compute exp(φ) for φ in so(n) using matrix exponential via series.

    exp(A) = I + A + A²/2! + A³/3! + ...

    For small ||φ||, truncated series is accurate and efficient.

    Args:
        phi: (..., dim_g) Lie algebra coefficients
        generators: (dim_g, n, n) basis matrices
        max_terms: Number of series terms

    Returns:
        R: (..., n, n) rotation matrices in SO(N)
    """
    # Construct matrix A = φ^a T_a
    # phi: (..., dim_g), generators: (dim_g, n, n)
    A = torch.einsum('...a,aij->...ij', phi, generators)

    n = generators.shape[1]
    batch_shape = phi.shape[:-1]

    # Initialize: exp(A) = I
    I = torch.eye(n, device=phi.device, dtype=phi.dtype)
    I = I.expand(*batch_shape, n, n)

    result = I.clone()
    A_power = I.clone()  # A^0 = I

    for k in range(1, max_terms):
        A_power = torch.einsum('...ij,...jk->...ik', A_power, A) / k
        result = result + A_power

    return result


def rodrigues_so3(phi: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues formula for SO(3) - efficient closed form.

    exp(φ) = I + (sin θ / θ) A + ((1 - cos θ) / θ²) A²

    where θ = ||φ|| and A is the Lie algebra element (skew-symmetric matrix).

    IMPORTANT: Uses the same generator basis as so_n_generators(3):
        T_0: (0,1)=1, (1,0)=-1  (xy-plane rotation)
        T_1: (0,2)=1, (2,0)=-1  (xz-plane rotation)
        T_2: (1,2)=1, (2,1)=-1  (yz-plane rotation)

    So A = φ_0 T_0 + φ_1 T_1 + φ_2 T_2 gives:
        A = [[0, φ_0, φ_1], [-φ_0, 0, φ_2], [-φ_1, -φ_2, 0]]

    Only valid for SO(3) (dim_g = 3)!

    Args:
        phi: (..., 3) Lie algebra coefficients

    Returns:
        R: (..., 3, 3) rotation matrix
    """
    theta = torch.norm(phi, dim=-1, keepdim=True).unsqueeze(-1)  # (..., 1, 1)
    theta = theta.clamp(min=1e-8)  # Avoid division by zero

    # Skew-symmetric matrix A = φ^a T_a matching so_n_generators(3) basis
    # A = [[0, φ_0, φ_1], [-φ_0, 0, φ_2], [-φ_1, -φ_2, 0]]
    batch_shape = phi.shape[:-1]
    A = torch.zeros(*batch_shape, 3, 3, device=phi.device, dtype=phi.dtype)
    A[..., 0, 1] = phi[..., 0]
    A[..., 0, 2] = phi[..., 1]
    A[..., 1, 0] = -phi[..., 0]
    A[..., 1, 2] = phi[..., 2]
    A[..., 2, 0] = -phi[..., 1]
    A[..., 2, 1] = -phi[..., 2]

    I = torch.eye(3, device=phi.device, dtype=phi.dtype).expand(*batch_shape, 3, 3)

    # Rodrigues formula
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    R = I + (sin_theta / theta) * A + \
        ((1 - cos_theta) / (theta ** 2)) * torch.einsum('...ij,...jk->...ik', A, A)

    return R


# =============================================================================
# PART 3: GAUSSIAN BELIEFS WITH BLOCK-DIAGONAL COVARIANCE
# =============================================================================

class IrrepSpec:
    """
    Specification of irreducible representations.

    Example: [('fund', 3, 10)] means 3 copies of 10-dim fundamental rep
    Total dim = 3 * 10 = 30
    """
    def __init__(self, specs: List[Tuple[str, int, int]]):
        """
        Args:
            specs: List of (name, multiplicity, dimension) tuples
        """
        self.specs = specs
        self.block_dims = []
        self.block_starts = []

        offset = 0
        for name, mult, dim in specs:
            for _ in range(mult):
                self.block_starts.append(offset)
                self.block_dims.append(dim)
                offset += dim

        self.total_dim = offset
        self.n_blocks = len(self.block_dims)

    def __repr__(self):
        return f"IrrepSpec({self.specs}, total_dim={self.total_dim})"


class BlockDiagonalCovariance:
    """
    Block-diagonal covariance matrix aligned with irrep structure.

    Stores r blocks of sizes (d_1, d_1), (d_2, d_2), ..., (d_r, d_r)
    Memory: O(Σ d_i²) instead of O((Σ d_i)²)

    For embed_dim=30 with 3×SO(10):
        - Full: 30×30 = 900 params
        - Block-diagonal: 3×(10×10) = 300 params
        - Diagonal: 30 params (loses intra-block correlations)
    """

    def __init__(self, blocks: List[torch.Tensor]):
        """
        Args:
            blocks: List of (..., d_i, d_i) covariance matrices
        """
        self.blocks = blocks
        self.n_blocks = len(blocks)
        self.block_dims = [b.shape[-1] for b in blocks]
        self.total_dim = sum(self.block_dims)

    @classmethod
    def from_diagonal(cls, diag: torch.Tensor, irrep_spec: IrrepSpec) -> 'BlockDiagonalCovariance':
        """Create block-diagonal from diagonal (scalar per dimension)."""
        blocks = []
        for start, dim in zip(irrep_spec.block_starts, irrep_spec.block_dims):
            # Extract diagonal for this block, make it a diagonal matrix
            block_diag = diag[..., start:start+dim]
            block = torch.diag_embed(block_diag)
            blocks.append(block)
        return cls(blocks)

    @classmethod
    def from_scalar_per_block(cls, scalars: torch.Tensor, irrep_spec: IrrepSpec) -> 'BlockDiagonalCovariance':
        """Create isotropic blocks: Σ_i = σ_i² I."""
        blocks = []
        for i, dim in enumerate(irrep_spec.block_dims):
            # scalars[..., i] is variance for block i
            var = scalars[..., i:i+1, None]  # (..., 1, 1)
            I = torch.eye(dim, device=scalars.device, dtype=scalars.dtype)
            block = var * I
            blocks.append(block)
        return cls(blocks)

    def to_full(self) -> torch.Tensor:
        """Convert to full (K, K) matrix. Use sparingly!"""
        batch_shape = self.blocks[0].shape[:-2]
        full = torch.zeros(*batch_shape, self.total_dim, self.total_dim,
                          device=self.blocks[0].device, dtype=self.blocks[0].dtype)
        offset = 0
        for block in self.blocks:
            d = block.shape[-1]
            full[..., offset:offset+d, offset:offset+d] = block
            offset += d
        return full

    def log_det(self) -> torch.Tensor:
        """Compute log|Σ| = Σ_i log|Σ_i|."""
        log_det = 0
        for block in self.blocks:
            # For each block, compute log determinant
            log_det = log_det + torch.linalg.slogdet(block)[1]
        return log_det

    def solve(self, v: torch.Tensor, irrep_spec: IrrepSpec) -> torch.Tensor:
        """Compute Σ⁻¹ v block-wise."""
        result = torch.zeros_like(v)
        for i, (start, dim) in enumerate(zip(irrep_spec.block_starts, irrep_spec.block_dims)):
            v_block = v[..., start:start+dim]
            # Solve Σ_i x = v_block
            solved = torch.linalg.solve(self.blocks[i], v_block.unsqueeze(-1)).squeeze(-1)
            result[..., start:start+dim] = solved
        return result

    def trace_solve(self, other: 'BlockDiagonalCovariance') -> torch.Tensor:
        """Compute tr(Σ⁻¹ other) = Σ_i tr(Σ_i⁻¹ other_i)."""
        trace = 0
        for my_block, other_block in zip(self.blocks, other.blocks):
            # tr(A⁻¹ B) = tr(solve(A, B))
            solved = torch.linalg.solve(my_block, other_block)
            trace = trace + torch.diagonal(solved, dim1=-2, dim2=-1).sum(dim=-1)
        return trace

    def quadratic_form(self, v: torch.Tensor, irrep_spec: IrrepSpec) -> torch.Tensor:
        """Compute v^T Σ⁻¹ v block-wise."""
        result = 0
        for i, (start, dim) in enumerate(zip(irrep_spec.block_starts, irrep_spec.block_dims)):
            v_block = v[..., start:start+dim]
            # v^T Σ⁻¹ v for this block
            solved = torch.linalg.solve(self.blocks[i], v_block.unsqueeze(-1)).squeeze(-1)
            result = result + (v_block * solved).sum(dim=-1)
        return result

    def transform(self, R: torch.Tensor, irrep_spec: IrrepSpec) -> 'BlockDiagonalCovariance':
        """
        Apply rotation: Σ' = R Σ R^T, block-wise.

        R should be block-diagonal matching irrep structure.
        """
        new_blocks = []
        for i, (start, dim) in enumerate(zip(irrep_spec.block_starts, irrep_spec.block_dims)):
            R_block = R[..., start:start+dim, start:start+dim]
            # Σ'_i = R_i Σ_i R_i^T
            transformed = torch.einsum('...ij,...jk,...lk->...il',
                                       R_block, self.blocks[i], R_block)
            new_blocks.append(transformed)
        return BlockDiagonalCovariance(new_blocks)


@dataclass
class GaussianBelief:
    """
    Gaussian belief q = N(μ, Σ) with optional gauge frame φ.

    Supports three covariance modes:
    1. Diagonal: sigma is (..., K) - fastest, loses all correlations
    2. Block-diagonal: sigma is BlockDiagonalCovariance - preserves intra-irrep correlations
    3. Full: sigma is (..., K, K) - slowest, full correlations
    """
    mu: torch.Tensor           # (..., K) mean
    sigma: torch.Tensor        # (..., K) diagonal, (..., K, K) full, or BlockDiagonalCovariance
    phi: Optional[torch.Tensor] = None  # (..., dim_g) gauge frame
    irrep_spec: Optional[IrrepSpec] = None  # For block-diagonal mode

    @property
    def is_diagonal(self) -> bool:
        return isinstance(self.sigma, torch.Tensor) and self.sigma.dim() == self.mu.dim()

    @property
    def is_block_diagonal(self) -> bool:
        return isinstance(self.sigma, BlockDiagonalCovariance)

    @property
    def is_full(self) -> bool:
        return isinstance(self.sigma, torch.Tensor) and self.sigma.dim() == self.mu.dim() + 1


def kl_divergence_gaussian(q: GaussianBelief, p: GaussianBelief,
                           transport: Optional[torch.Tensor] = None,
                           irrep_spec: Optional[IrrepSpec] = None) -> torch.Tensor:
    """
    KL divergence KL(q || p) between Gaussians, with optional transport.

    If transport Ω is provided, we compute KL(q || Ω·p) where:
        Ω·p = N(Ω μ_p, Ω Σ_p Ω^T)

    Supports diagonal, block-diagonal, and full covariance modes.

    KL = ½[tr(Σ_p⁻¹ Σ_q) + (μ_p - μ_q)^T Σ_p⁻¹ (μ_p - μ_q) - K + log|Σ_p|/|Σ_q|]

    Args:
        q: Query belief N(μ_q, Σ_q)
        p: Prior/target belief N(μ_p, Σ_p)
        transport: Optional (..., K, K) rotation matrix
        irrep_spec: Required for block-diagonal mode

    Returns:
        kl: (...,) KL divergence values
    """
    mu_q = q.mu
    mu_p = p.mu
    sigma_q = q.sigma
    sigma_p = p.sigma

    K = mu_q.shape[-1]

    # Apply transport if provided
    if transport is not None:
        # Rotate prior mean: μ_p' = Ω μ_p
        mu_p = torch.einsum('...ij,...j->...i', transport, mu_p)

        # Transform covariance based on type
        if q.is_block_diagonal and irrep_spec is not None:
            sigma_p = sigma_p.transform(transport, irrep_spec)
        elif not q.is_diagonal:
            # Full covariance: Σ_p' = Ω Σ_p Ω^T
            sigma_p = torch.einsum('...ij,...jk,...lk->...il',
                                   transport, sigma_p, transport)
        # For diagonal, assume transport preserves diagonal (block structure)

    # Compute KL based on covariance type
    if q.is_diagonal and (isinstance(sigma_p, torch.Tensor) and sigma_p.dim() == mu_p.dim()):
        # Both diagonal - most efficient
        var_q = sigma_q
        var_p = sigma_p

        var_ratio = var_q / (var_p + 1e-8)
        diff = mu_p - mu_q
        mahal = diff ** 2 / (var_p + 1e-8)
        log_det_ratio = torch.log(var_p + 1e-8) - torch.log(var_q + 1e-8)

        kl = 0.5 * (var_ratio + mahal - 1 + log_det_ratio).sum(dim=-1)

    elif q.is_block_diagonal and isinstance(sigma_p, BlockDiagonalCovariance):
        # Both block-diagonal - efficient block-wise computation
        assert irrep_spec is not None, "irrep_spec required for block-diagonal KL"

        # tr(Σ_p⁻¹ Σ_q)
        trace_term = sigma_p.trace_solve(sigma_q)

        # (μ_p - μ_q)^T Σ_p⁻¹ (μ_p - μ_q)
        diff = mu_p - mu_q
        mahal_term = sigma_p.quadratic_form(diff, irrep_spec)

        # log|Σ_p| - log|Σ_q|
        log_det_p = sigma_p.log_det()
        log_det_q = sigma_q.log_det()
        log_det_ratio = log_det_p - log_det_q

        kl = 0.5 * (trace_term + mahal_term - K + log_det_ratio)

    elif q.is_full or (isinstance(sigma_p, torch.Tensor) and sigma_p.dim() == mu_p.dim() + 1):
        # Full covariance
        # tr(Σ_p⁻¹ Σ_q)
        Sigma_p_inv_Sigma_q = torch.linalg.solve(sigma_p, sigma_q)
        trace_term = torch.diagonal(Sigma_p_inv_Sigma_q, dim1=-2, dim2=-1).sum(dim=-1)

        # (μ_p - μ_q)^T Σ_p⁻¹ (μ_p - μ_q)
        diff = (mu_p - mu_q).unsqueeze(-1)
        mahal_term = torch.linalg.solve(sigma_p, diff).squeeze(-1)
        mahal_term = (diff.squeeze(-1) * mahal_term).sum(dim=-1)

        # log|Σ_p| - log|Σ_q|
        log_det_p = torch.linalg.slogdet(sigma_p)[1]
        log_det_q = torch.linalg.slogdet(sigma_q)[1]
        log_det_ratio = log_det_p - log_det_q

        kl = 0.5 * (trace_term + mahal_term - K + log_det_ratio)

    else:
        raise ValueError(f"Incompatible covariance types: q={type(sigma_q)}, p={type(sigma_p)}")

    return kl


# =============================================================================
# PART 4: VARIATIONAL FREE ENERGY FUNCTIONAL
# =============================================================================

class VFEFunctional(nn.Module):
    """
    The Variational Free Energy functional.

    F[q] = α·F_self + β·F_align + γ·F_prior + F_obs

    where:
    - F_self = Σ_i H[q_i] (entropy / uncertainty cost)
    - F_align = Σ_{i,j} w_ij · KL(q_i || Ω_ij q_j) (belief alignment)
    - F_prior = Σ_i KL(q_i || π_i) (prior divergence)
    - F_obs = Σ_i E_q[-log p(y_i | z_i)] (observation likelihood)
    """

    def __init__(self,
                 embed_dim: int,
                 gauge_dim: int,  # N for SO(N)
                 irrep_spec: Optional[IrrepSpec] = None,
                 alpha: float = 1.0,   # Self-coupling weight
                 beta: float = 1.0,    # Alignment weight
                 gamma: float = 0.1,   # Prior weight
                 bch_order: int = 2,   # BCH truncation
                 temperature: float = 1.0):  # Attention temperature
        super().__init__()

        self.embed_dim = embed_dim
        self.gauge_dim = gauge_dim
        self.irrep_spec = irrep_spec
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bch_order = bch_order
        self.temperature = temperature

        # Precompute SO(N) structure
        self.register_buffer('generators', so_n_generators(gauge_dim))
        self.register_buffer('f_abc', so_n_structure_constants(gauge_dim))

        self.dim_g = gauge_dim * (gauge_dim - 1) // 2  # Lie algebra dimension

        # For multiple irreps, we need block-diagonal generators
        if irrep_spec is not None:
            self._build_block_diagonal_generators(irrep_spec)

    def _build_block_diagonal_generators(self, irrep_spec: IrrepSpec):
        """
        Build block-diagonal generators for multiple irreps.

        Each block gets its own copy of the SO(N) generators.
        Total embedding dim = sum of block dims.
        """
        # For now, assume all blocks are the same gauge_dim (fundamental rep)
        # Each block of size d gets d(d-1)/2 generators
        K = irrep_spec.total_dim
        n_blocks = irrep_spec.n_blocks

        # Build block-diagonal generators
        # Shape: (dim_g, K, K) where generators are block-diagonal
        block_gens = torch.zeros(self.dim_g, K, K)

        for i, (start, dim) in enumerate(zip(irrep_spec.block_starts, irrep_spec.block_dims)):
            # Only include if this block matches gauge_dim
            if dim == self.gauge_dim:
                block_gens[:, start:start+dim, start:start+dim] = self.generators

        self.register_buffer('block_generators', block_gens)

    def compute_transport(self, phi_i: torch.Tensor, phi_j: torch.Tensor) -> torch.Tensor:
        """
        Compute transport operator Ω_ij = exp(BCH(φ_i, -φ_j)).

        Args:
            phi_i: (B, N, dim_g) source frames
            phi_j: (B, N, dim_g) target frames

        Returns:
            omega: (B, N, N, gauge_dim, gauge_dim) transport matrices
        """
        B, N, dim_g = phi_i.shape

        # Expand for pairwise computation: (B, N, 1, dim_g) vs (B, 1, N, dim_g)
        phi_i_exp = phi_i.unsqueeze(2)  # (B, N, 1, dim_g)
        phi_j_exp = phi_j.unsqueeze(1)  # (B, 1, N, dim_g)

        # BCH combination: φ_ij for all pairs
        phi_ij = bch_combine(phi_i_exp, phi_j_exp, self.f_abc, order=self.bch_order)
        # Shape: (B, N, N, dim_g)

        # Exponentiate to get rotation matrices
        if self.gauge_dim == 3:
            # Use efficient Rodrigues formula for SO(3)
            omega = rodrigues_so3(phi_ij)
        else:
            # General SO(N) via series
            omega = exp_so_n(phi_ij, self.generators)

        return omega  # (B, N, N, gauge_dim, gauge_dim)

    def f_self(self, beliefs: GaussianBelief) -> torch.Tensor:
        """
        Self-coupling term: negative entropy (uncertainty cost).

        H[q] = ½ log|2πe Σ| = ½(K log(2πe) + log|Σ|)

        For diagonal: H = ½ Σ_k log(2πe σ_k)
        For block-diagonal: H = ½ Σ_b log|2πe Σ_b|
        """
        if beliefs.is_diagonal:
            # Diagonal case: sum of log variances
            log_det = torch.log(beliefs.sigma + 1e-8).sum(dim=-1)
        elif beliefs.is_block_diagonal:
            # Block-diagonal: sum of block log determinants
            log_det = beliefs.sigma.log_det()
        else:
            # Full covariance: log determinant
            log_det = torch.linalg.slogdet(beliefs.sigma)[1]

        K = beliefs.mu.shape[-1]
        entropy = 0.5 * (K * math.log(2 * math.pi * math.e) + log_det)

        # Return negative entropy (we minimize F, high entropy is good)
        return -entropy.mean()

    def f_align(self, beliefs: GaussianBelief,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Belief alignment term: pairwise KL with transport.

        F_align = Σ_{i,j} w_ij · KL(q_i || Ω_ij q_j)

        Returns both the free energy and the attention weights (for output).

        For block-diagonal covariance with multiple irreps:
        - Transport is block-diagonal: Ω = diag(Ω_1, Ω_2, ..., Ω_r)
        - Each block transforms independently
        """
        B, N, K = beliefs.mu.shape

        # Compute transport operators (in the gauge_dim space)
        omega_small = self.compute_transport(beliefs.phi, beliefs.phi)
        # Shape: (B, N, N, gauge_dim, gauge_dim)

        # Expand to full embed_dim if using multiple irreps
        if self.irrep_spec is not None and hasattr(self, 'block_generators'):
            # Build block-diagonal transport in full K×K space
            omega = torch.zeros(B, N, N, K, K, device=beliefs.mu.device, dtype=beliefs.mu.dtype)
            for i, (start, dim) in enumerate(zip(self.irrep_spec.block_starts,
                                                   self.irrep_spec.block_dims)):
                if dim == self.gauge_dim:
                    omega[..., start:start+dim, start:start+dim] = omega_small
                else:
                    # Identity for non-matching blocks
                    omega[..., start:start+dim, start:start+dim] = torch.eye(
                        dim, device=omega.device, dtype=omega.dtype)
        else:
            omega = omega_small

        # Expand beliefs for pairwise computation
        mu_i = beliefs.mu.unsqueeze(2)      # (B, N, 1, K)
        mu_j = beliefs.mu.unsqueeze(1)      # (B, 1, N, K)

        # Transport μ_j: Ω_ij μ_j
        mu_j_transported = torch.einsum('bnmij,bnmj->bnmi', omega, mu_j)

        if beliefs.is_diagonal:
            sigma_i = beliefs.sigma.unsqueeze(2)  # (B, N, 1, K)
            sigma_j = beliefs.sigma.unsqueeze(1)  # (B, 1, N, K)

            # For diagonal covariance, rotation preserves diagonal if isotropic per block
            # Approximation: variance is preserved under rotation
            sigma_j_transported = sigma_j

            # KL(q_i || Ω_ij q_j) - diagonal case
            var_ratio = sigma_i / (sigma_j_transported + 1e-8)
            diff = mu_j_transported - mu_i
            mahal = diff ** 2 / (sigma_j_transported + 1e-8)
            log_det_ratio = torch.log(sigma_j_transported + 1e-8) - torch.log(sigma_i + 1e-8)

            kl_ij = 0.5 * (var_ratio + mahal - 1 + log_det_ratio).sum(dim=-1)

        elif beliefs.is_block_diagonal:
            # Block-diagonal case: compute KL block by block
            # This is more expensive but preserves intra-block correlations
            kl_ij = torch.zeros(B, N, N, device=beliefs.mu.device, dtype=beliefs.mu.dtype)

            for b_idx, (start, dim) in enumerate(zip(self.irrep_spec.block_starts,
                                                      self.irrep_spec.block_dims)):
                # Extract block data
                mu_i_block = mu_i[..., start:start+dim]
                mu_j_block = mu_j_transported[..., start:start+dim]
                sigma_i_block = beliefs.sigma.blocks[b_idx].unsqueeze(2)  # (B, N, 1, d, d)
                sigma_j_block = beliefs.sigma.blocks[b_idx].unsqueeze(1)  # (B, 1, N, d, d)

                # Transform sigma_j block
                omega_block = omega[..., start:start+dim, start:start+dim]
                sigma_j_block = torch.einsum('bnmij,bnmjk,bnmlk->bnmil',
                                             omega_block, sigma_j_block, omega_block)

                # KL for this block
                diff = mu_j_block - mu_i_block
                # tr(Σ_i⁻¹ Σ_j)
                trace_term = torch.linalg.solve(sigma_i_block, sigma_j_block)
                trace_term = torch.diagonal(trace_term, dim1=-2, dim2=-1).sum(dim=-1)
                # (μ_j - μ_i)^T Σ_i⁻¹ (μ_j - μ_i)
                mahal = torch.linalg.solve(sigma_i_block, diff.unsqueeze(-1)).squeeze(-1)
                mahal = (diff * mahal).sum(dim=-1)
                # log|Σ_i| - log|Σ_j|
                log_det_i = torch.linalg.slogdet(sigma_i_block)[1]
                log_det_j = torch.linalg.slogdet(sigma_j_block)[1]

                kl_ij = kl_ij + 0.5 * (trace_term + mahal - dim + log_det_i - log_det_j)

        else:
            # Full covariance case
            raise NotImplementedError("Full covariance f_align not implemented")

        # Convert to attention weights with temperature scaling
        # Lower KL = higher attention
        attn_logits = -kl_ij / self.temperature

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=-1)

        # Free energy: expected KL under attention
        f_align = (attn_weights * kl_ij).sum(dim=-1).mean()

        return f_align, attn_weights

    def f_prior(self, beliefs: GaussianBelief,
                priors: GaussianBelief) -> torch.Tensor:
        """
        Prior coupling: KL(q || π).
        """
        kl = kl_divergence_gaussian(beliefs, priors)
        return kl.mean()

    def f_obs(self, beliefs: GaussianBelief,
              targets: torch.Tensor,
              output_proj: nn.Linear) -> torch.Tensor:
        """
        Observation term: cross-entropy prediction loss.

        E_q[-log p(y | z)] ≈ -log p(y | μ_q)

        Using point estimate at mean for efficiency.
        """
        # Project belief means to vocabulary
        logits = output_proj(beliefs.mu)  # (B, N, vocab_size)

        # Cross-entropy loss (use reshape for non-contiguous tensors)
        B, N, V = logits.shape
        loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction='mean')

        return loss

    def forward(self, beliefs: GaussianBelief,
                priors: GaussianBelief,
                targets: torch.Tensor,
                output_proj: nn.Linear,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Compute full VFE.

        Returns:
            vfe: Total free energy
            components: Dict of individual terms
        """
        f_self = self.f_self(beliefs)
        f_align, attn = self.f_align(beliefs, mask)
        f_prior = self.f_prior(beliefs, priors)
        f_obs = self.f_obs(beliefs, targets, output_proj)

        vfe = (self.alpha * f_self +
               self.beta * f_align +
               self.gamma * f_prior +
               f_obs)

        components = {
            'f_self': f_self.item(),
            'f_align': f_align.item(),
            'f_prior': f_prior.item(),
            'f_obs': f_obs.item(),
            'attention': attn,
        }

        return vfe, components


# =============================================================================
# PART 5: BELIEF EMBEDDINGS WITH HAAR INITIALIZATION
# =============================================================================

class BeliefEmbedding(nn.Module):
    """
    Learnable belief embeddings: μ, Σ, φ for each token.

    Each token t has a belief q_t = N(μ_t, Σ_t) in gauge frame φ_t.

    Initialization:
    - μ: Xavier/Glorot for good gradient flow
    - Σ: Small positive values (log-parameterized)
    - φ: Haar measure on SO(N) for symmetry breaking
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 gauge_dim: int,
                 init_sigma: float = 1.0,
                 init_phi_scale: float = 0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.gauge_dim = gauge_dim
        self.dim_g = gauge_dim * (gauge_dim - 1) // 2

        # Mean embeddings μ
        self.mu = nn.Embedding(vocab_size, embed_dim)

        # Log-variance embeddings (ensures positivity)
        self.log_sigma = nn.Embedding(vocab_size, embed_dim)

        # Gauge frame embeddings φ (Lie algebra coefficients)
        self.phi = nn.Embedding(vocab_size, self.dim_g)

        self._init_weights(init_sigma, init_phi_scale)

    def _init_weights(self, init_sigma: float, init_phi_scale: float):
        """Initialize embeddings."""
        # μ: Xavier initialization
        nn.init.xavier_uniform_(self.mu.weight)

        # Σ: Initialize to give desired variance
        nn.init.constant_(self.log_sigma.weight, math.log(init_sigma))

        # φ: Haar initialization on SO(N)
        # For small angles, Haar measure ≈ uniform on Lie algebra
        # Scale controls typical rotation angle
        nn.init.uniform_(self.phi.weight, -init_phi_scale, init_phi_scale)

    def forward(self, token_ids: torch.Tensor) -> GaussianBelief:
        """
        Look up beliefs for tokens.

        Args:
            token_ids: (B, N) token indices

        Returns:
            beliefs: GaussianBelief with μ, σ, φ
        """
        mu = self.mu(token_ids)
        sigma = torch.exp(self.log_sigma(token_ids))  # Ensure positive
        phi = self.phi(token_ids)

        return GaussianBelief(mu=mu, sigma=sigma, phi=phi)


# =============================================================================
# PART 6: Q-FLOW (BELIEF DYNAMICS)
# =============================================================================

class QFlow(nn.Module):
    """
    Q-flow: Fast belief updates via natural gradient descent on VFE.

    dq/dt = -η · F̃⁻¹ · ∇_q F

    where F̃ is the Fisher information metric.

    For Gaussian beliefs, natural gradient has closed form:
    - ∇̃_μ F = ∇_μ F (Fisher metric is identity for mean)
    - ∇̃_Σ F = Σ · ∇_Σ F · Σ (Fisher metric for covariance)
    """

    def __init__(self,
                 embed_dim: int,
                 n_iterations: int = 1,
                 mu_lr: float = 0.1,
                 sigma_lr: float = 0.01):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_iterations = n_iterations

        # Learnable learning rates (can adapt during training)
        self.mu_lr = nn.Parameter(torch.tensor(mu_lr))
        self.sigma_lr = nn.Parameter(torch.tensor(sigma_lr))

    def step(self, beliefs: GaussianBelief,
             grad_mu: torch.Tensor,
             grad_sigma: torch.Tensor) -> GaussianBelief:
        """
        Single natural gradient step.

        Args:
            beliefs: Current beliefs
            grad_mu: Gradient w.r.t. μ
            grad_sigma: Gradient w.r.t. σ (for diagonal parameterization)

        Returns:
            Updated beliefs
        """
        # Clamp gradients to prevent explosion
        grad_mu = torch.clamp(grad_mu, -10.0, 10.0)
        grad_sigma = torch.clamp(grad_sigma, -10.0, 10.0)

        # Natural gradient for mean (Fisher = I)
        new_mu = beliefs.mu - self.mu_lr * grad_mu

        # Natural gradient for variance
        # For diagonal Σ, Fisher metric gives: ∇̃_σ = σ² · ∇_σ
        # Update in log-space for stability
        log_sigma = torch.log(beliefs.sigma + 1e-8)
        new_log_sigma = log_sigma - self.sigma_lr * beliefs.sigma * grad_sigma
        new_log_sigma = torch.clamp(new_log_sigma, -10.0, 10.0)  # Prevent exp overflow
        new_sigma = torch.exp(new_log_sigma)

        # Clamp sigma to reasonable range
        new_sigma = torch.clamp(new_sigma, 1e-6, 1e6)

        return GaussianBelief(mu=new_mu, sigma=new_sigma, phi=beliefs.phi)


# =============================================================================
# PART 7: P-FLOW (PRIOR DYNAMICS)
# =============================================================================

class PFlow(nn.Module):
    """
    P-flow: Slow prior updates toward successful beliefs.

    π_t+1 = (1 - η) · π_t + η · EMA(q_successful)

    "Successful" beliefs are those that achieved low VFE.
    """

    def __init__(self,
                 embed_dim: int,
                 ema_decay: float = 0.99):
        super().__init__()

        self.embed_dim = embed_dim
        self.ema_decay = ema_decay

    def update(self, priors: GaussianBelief,
               beliefs: GaussianBelief,
               success_weights: Optional[torch.Tensor] = None) -> GaussianBelief:
        """
        Update priors toward successful beliefs.

        Args:
            priors: Current priors
            beliefs: Current beliefs (after Q-flow)
            success_weights: Optional weights indicating belief success

        Returns:
            Updated priors
        """
        if success_weights is None:
            # Uniform weighting
            target_mu = beliefs.mu.mean(dim=1, keepdim=True).expand_as(priors.mu)
            target_sigma = beliefs.sigma.mean(dim=1, keepdim=True).expand_as(priors.sigma)
        else:
            # Weighted average
            weights = success_weights.unsqueeze(-1)
            target_mu = (beliefs.mu * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True)
            target_sigma = (beliefs.sigma * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True)

        # EMA update
        new_mu = self.ema_decay * priors.mu + (1 - self.ema_decay) * target_mu
        new_sigma = self.ema_decay * priors.sigma + (1 - self.ema_decay) * target_sigma

        return GaussianBelief(mu=new_mu, sigma=new_sigma, phi=priors.phi)


# =============================================================================
# PART 8: FULL FEP TRANSFORMER
# =============================================================================

class FEPTransformer(nn.Module):
    """
    Full Free Energy Principle Transformer.

    Architecture:
    1. Token → Belief embedding (μ, Σ, φ)
    2. Q-flow: Minimize VFE via natural gradient (attention emerges)
    3. P-flow: Update priors toward successful beliefs
    4. Output: Project final beliefs to vocabulary

    Multi-layer: Stack Q-flow layers for hierarchical belief refinement.
    Each layer has its own priors and Q-flow dynamics.

    NO learned attention weights, NO MLP layers.
    Everything emerges from VFE minimization.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 gauge_dim: int,
                 irrep_spec: Optional[List[Tuple[str, int, int]]] = None,
                 n_layers: int = 1,
                 n_q_iterations: int = 5,
                 alpha: float = 0.1,
                 beta: float = 1.0,
                 gamma: float = 0.1,
                 bch_order: int = 2,
                 temperature: float = 1.0,
                 tie_embeddings: bool = False,
                 residual: bool = True,
                 observe_during_qflow: bool = False,
                 blind_iterations: int = 3,
                 lambda_obs: float = 1.0):
        """
        Args:
            observe_during_qflow: If True, include observation term (f_obs) in VFE
                during Q-flow iterations. Two-phase: first blind_iterations are
                alignment-only, then observations are added.
            blind_iterations: Number of Q-flow iterations for alignment before
                adding observations. Only used when observe_during_qflow=True.
                E.g., if n_q_iterations=5 and blind_iterations=3:
                  - Iterations 0,1,2: alignment only (words interact)
                  - Iterations 3,4: add observations (check prediction)
            lambda_obs: Weight for observation term in VFE (default 1.0).
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.gauge_dim = gauge_dim
        self.n_layers = n_layers
        self.residual = residual
        self.observe_during_qflow = observe_during_qflow
        self.blind_iterations = blind_iterations
        self.lambda_obs = lambda_obs

        # Parse irrep specification
        if irrep_spec is not None:
            self.irrep_spec = IrrepSpec(irrep_spec)
            assert self.irrep_spec.total_dim == embed_dim, \
                f"Irrep spec total dim {self.irrep_spec.total_dim} != embed_dim {embed_dim}"
        else:
            # Default: single block matching gauge_dim
            self.irrep_spec = None

        # Belief embeddings (input layer)
        self.belief_embed = BeliefEmbedding(vocab_size, embed_dim, gauge_dim)

        # Per-layer priors and Q-flow dynamics
        self.layer_priors = nn.ModuleList([
            BeliefEmbedding(vocab_size, embed_dim, gauge_dim)
            for _ in range(n_layers)
        ])

        self.layer_vfe = nn.ModuleList([
            VFEFunctional(embed_dim, gauge_dim, self.irrep_spec,
                         alpha, beta, gamma, bch_order, temperature)
            for _ in range(n_layers)
        ])

        self.layer_q_flow = nn.ModuleList([
            QFlow(embed_dim, n_iterations=n_q_iterations)
            for _ in range(n_layers)
        ])

        # P-flow dynamics (shared across layers)
        self.p_flow = PFlow(embed_dim)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)

        if tie_embeddings:
            self.output_proj.weight = self.belief_embed.mu.weight

    def _run_q_flow_layer(self, beliefs: GaussianBelief, priors: GaussianBelief,
                          mask: torch.Tensor, layer_idx: int,
                          targets: Optional[torch.Tensor] = None) -> Tuple[GaussianBelief, torch.Tensor, torch.Tensor]:
        """
        Run Q-flow iterations for a single layer.

        Two-phase Q-flow (when observe_during_qflow=True):
          Phase 1 (iterations 0 to blind_iterations-1): Alignment only
            - Words interact via KL, build contextual representation
          Phase 2 (iterations blind_iterations onwards): Add observations
            - Check prediction against target, refine beliefs

        This mimics human reading: understand first, then predict.

        Returns updated beliefs, final vfe_internal, and attention weights.
        """
        vfe = self.layer_vfe[layer_idx]
        q_flow = self.layer_q_flow[layer_idx]

        for q_iter in range(q_flow.n_iterations):
            # Clamp sigma BEFORE computing VFE to prevent NaN
            beliefs = GaussianBelief(
                mu=beliefs.mu,
                sigma=torch.clamp(beliefs.sigma, 1e-4, 1e4),
                phi=beliefs.phi
            )

            # Ensure gradients can be computed
            beliefs.mu.requires_grad_(True)
            beliefs.sigma.requires_grad_(True)

            f_self = vfe.f_self(beliefs)
            f_align, attn = vfe.f_align(beliefs, mask)
            f_prior = vfe.f_prior(beliefs, priors)

            # Internal VFE (alignment phase)
            vfe_internal = (vfe.alpha * f_self +
                           vfe.beta * f_align +
                           vfe.gamma * f_prior)

            # Two-phase: only add observations AFTER blind alignment iterations
            in_observation_phase = q_iter >= self.blind_iterations
            if self.observe_during_qflow and targets is not None and in_observation_phase:
                logits = self.output_proj(beliefs.mu)
                f_obs = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    targets.reshape(-1),
                    ignore_index=50256
                )
                vfe_internal = vfe_internal + self.lambda_obs * f_obs

            # Compute gradients w.r.t. beliefs
            # NOTE: create_graph=False for speed. Gradients still flow through
            # belief updates because beliefs.mu is tracked.
            grads = torch.autograd.grad(
                vfe_internal,
                [beliefs.mu, beliefs.sigma],
                create_graph=False,  # Don't build graph through Q-flow iterations
                retain_graph=True
            )

            # Update beliefs via natural gradient
            beliefs = q_flow.step(beliefs, grads[0], grads[1])

        return beliefs, vfe_internal, attn

    def forward(self,
                input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                return_components: bool = False) -> dict:
        """
        Forward pass.

        Args:
            input_ids: (B, N) input token IDs
            targets: (B, N) target token IDs (for training)
            return_components: Whether to return VFE components

        Returns:
            Dict with logits, loss, and optionally components
        """
        B, N = input_ids.shape

        # Get initial beliefs from embeddings
        beliefs = self.belief_embed(input_ids)

        # Causal mask for autoregressive
        mask = torch.tril(torch.ones(N, N, device=input_ids.device)).unsqueeze(0)

        # Multi-layer Q-flow: stack belief refinement
        # If observe_during_qflow=True: beliefs update based on observations (pure FEP)
        # If observe_during_qflow=False: blind Q-flow, f_obs only in final loss (honest LM)
        inference_mode = not torch.is_grad_enabled()

        # Only pass targets to Q-flow if observe_during_qflow is enabled
        qflow_targets = targets if self.observe_during_qflow else None

        total_vfe = 0.0
        all_attns = []

        for layer_idx in range(self.n_layers):
            # Get layer-specific priors
            priors = self.layer_priors[layer_idx](input_ids)

            # Store input for residual connection
            if self.residual and layer_idx > 0:
                beliefs_in = beliefs

            # Run Q-flow for this layer
            if inference_mode:
                with torch.enable_grad():
                    beliefs, vfe_internal, attn = self._run_q_flow_layer(
                        beliefs, priors, mask, layer_idx, qflow_targets)
            else:
                beliefs, vfe_internal, attn = self._run_q_flow_layer(
                    beliefs, priors, mask, layer_idx, qflow_targets)

            # Residual connection: add input beliefs to output
            if self.residual and layer_idx > 0:
                beliefs = GaussianBelief(
                    mu=beliefs.mu + beliefs_in.mu,
                    sigma=beliefs.sigma,  # Don't add variances
                    phi=beliefs.phi
                )

            total_vfe += vfe_internal
            all_attns.append(attn)

        # Output logits (AFTER all Q-flow layers)
        logits = self.output_proj(beliefs.mu)

        # Components for logging
        components = {'attention': all_attns[-1], 'vfe_internal': total_vfe.item() if hasattr(total_vfe, 'item') else total_vfe}

        result = {'logits': logits}

        if targets is not None:
            # Compute observation loss AFTER Q-flow (no label leakage!)
            # Ignore padding tokens (GPT-2 uses eos_token=50256 as pad)
            f_obs = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=50256  # Don't count padding in loss
            )
            components['f_obs'] = f_obs.item()

            # Total VFE = internal terms (summed across layers) + observation
            # The internal terms (f_self, f_align, f_prior) were already optimized by Q-flow
            # f_obs provides the learning signal
            result['loss'] = total_vfe + f_obs
            result['ce_loss'] = f_obs

        if return_components:
            result['components'] = components
            result['attention'] = components.get('attention')

        return result

    def generate(self,
                 input_ids: torch.Tensor,
                 max_new_tokens: int = 50,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        Autoregressive generation.
        """
        for _ in range(max_new_tokens):
            # Get predictions for current sequence
            outputs = self.forward(input_ids)
            logits = outputs['logits'][:, -1, :] / temperature

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# =============================================================================
# PART 9: TESTING / SANITY CHECKS
# =============================================================================

def test_so_n_algebra():
    """Test SO(N) Lie algebra implementation."""
    print("Testing SO(N) Lie algebra...")

    for n in [3, 5, 10]:
        generators = so_n_generators(n)
        f_abc = so_n_structure_constants(n)

        n_gen = n * (n - 1) // 2
        assert generators.shape == (n_gen, n, n), f"Wrong generator shape for SO({n})"
        assert f_abc.shape == (n_gen, n_gen, n_gen), f"Wrong f_abc shape for SO({n})"

        # Check antisymmetry: f_abc = -f_bac
        assert torch.allclose(f_abc, -f_abc.transpose(0, 1), atol=1e-6), \
            f"Structure constants not antisymmetric for SO({n})"

        # Check Jacobi identity: f_abe f_ecd + f_bce f_ead + f_cae f_ebd = 0
        jacobi = torch.einsum('abe,ecd->abcd', f_abc, f_abc) + \
                 torch.einsum('bce,ead->abcd', f_abc, f_abc) + \
                 torch.einsum('cae,ebd->abcd', f_abc, f_abc)
        assert torch.allclose(jacobi, torch.zeros_like(jacobi), atol=1e-5), \
            f"Jacobi identity violated for SO({n})"

        print(f"  SO({n}): {n_gen} generators, Jacobi ✓")

    print("SO(N) algebra tests passed!\n")


def test_bch():
    """Test BCH formula."""
    print("Testing BCH formula...")

    n = 3  # SO(3)
    f_abc = so_n_structure_constants(n)

    # For small angles, BCH should be approximately additive
    phi_1 = torch.randn(3) * 0.1
    phi_2 = torch.randn(3) * 0.1

    bch_1 = bch_combine(phi_1, phi_2, f_abc, order=1)
    bch_2 = bch_combine(phi_1, phi_2, f_abc, order=2)

    # Order 1 should just be sum
    assert torch.allclose(bch_1, phi_1 - phi_2), "BCH order 1 should be simple sum"

    # Order 2 should have commutator correction
    bracket = lie_bracket(phi_1, phi_2, f_abc)
    expected = phi_1 - phi_2 - 0.5 * bracket
    assert torch.allclose(bch_2, expected), "BCH order 2 incorrect"

    print("  BCH formula tests passed!\n")


def test_rodrigues():
    """Test Rodrigues formula."""
    print("Testing Rodrigues formula...")

    # Small rotation
    phi = torch.tensor([0.1, 0.2, 0.3])
    R = rodrigues_so3(phi.unsqueeze(0)).squeeze(0)

    # Check orthogonality
    should_be_I = R @ R.T
    assert torch.allclose(should_be_I, torch.eye(3), atol=1e-5), "R not orthogonal"

    # Check determinant = 1
    det = torch.linalg.det(R)
    assert torch.allclose(det, torch.tensor(1.0), atol=1e-5), "det(R) != 1"

    # Compare with matrix exponential
    generators = so_n_generators(3)
    R_exp = exp_so_n(phi.unsqueeze(0), generators, max_terms=10).squeeze(0)
    assert torch.allclose(R, R_exp, atol=1e-4), "Rodrigues != matrix exp"

    print("  Rodrigues formula tests passed!\n")


if __name__ == '__main__':
    test_so_n_algebra()
    test_bch()
    test_rodrigues()

    print("Creating FEP Transformer...")
    model = FEPTransformer(
        vocab_size=1000,
        embed_dim=10,  # Small for testing
        gauge_dim=10,  # SO(10)
        n_layers=1,
        n_q_iterations=3,
    )

    # Test forward pass
    x = torch.randint(0, 1000, (2, 16))  # Batch of 2, seq len 16
    y = torch.randint(0, 1000, (2, 16))

    output = model(x, y, return_components=True)
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"CE Loss: {output['ce_loss'].item():.4f}")
    print(f"Attention shape: {output['attention'].shape}")

    print("\nAll tests passed!")
