"""
Standard Transformer Baseline
==============================

Vanilla transformer with dot-product attention for fair comparison with gauge model.

NO gauge theory, NO SO(3), NO KL divergence - just standard MHA.

Parameter-matched to gauge model (~5,334 params):
    - vocab_size: 256
    - embed_dim: 11
    - n_layers: 2
    - n_heads: 1 (for K=11, single head to save params)
    - hidden_dim: 44

Author: Baseline for ablation study
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class StandardMultiHeadAttention(nn.Module):
    """
    Standard dot-product multi-head attention.

    β_ij = softmax(Q_i @ K_j^T / sqrt(d_k))
    output_i = Σ_j β_ij @ V_j

    Compare to gauge model:
        β_ij = softmax(-KL(q_i || Ω_ij[q_j]) / κ)
    """

    def __init__(self, embed_dim: int, n_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        # Warn about pathologically small head dimensions
        if self.head_dim < 16:
            import warnings
            warnings.warn(
                f"head_dim={self.head_dim} is very small (embed_dim={embed_dim}, n_heads={n_heads}). "
                f"This severely limits model capacity. Recommended head_dim >= 32. "
                f"Consider using fewer heads (e.g., n_heads={embed_dim // 32} for head_dim=32).",
                UserWarning
            )

        # Standard attention projections (THIS IS WHAT GAUGE MODEL DOESN'T HAVE!)
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim) input
            mask: (N, N) or (B, N, N) causal mask
            return_attention: If True, also return attention weights

        Returns:
            (B, N, embed_dim) attended output, or tuple with attention weights
        """
        B, N, embed_dim = x.shape

        # Project to Q, K, V
        Q = self.W_Q(x)  # (B, N, embed_dim)
        K = self.W_K(x)  # (B, N, embed_dim)
        V = self.W_V(x)  # (B, N, embed_dim)

        # Reshape for multi-head (if n_heads > 1)
        Q = Q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        K = K.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        # Apply causal mask
        if mask is not None:
            # Ensure mask has shape (1, 1, N, N) for proper broadcasting with (B, H, N, N) scores
            while mask.dim() < 4:
                mask = mask.unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # (B, H, N, N)
        attn_weights_dropped = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights_dropped, V)  # (B, H, N, D)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, N, embed_dim)  # (B, N, embed_dim)

        # Output projection
        out = self.W_O(out)

        if return_attention:
            return out, attn_weights  # Return pre-dropout weights for visualization
        return out


class StandardFFN(nn.Module):
    """
    Standard feed-forward network.

    FFN(x) = W2 @ GELU(W1 @ x + b1) + b2

    Compare to gauge model's variational FFN:
        Performs inference via gradient descent on free energy
    """

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class StandardTransformerBlock(nn.Module):
    """
    Standard transformer block: LayerNorm -> MHA -> Add -> LayerNorm -> FFN -> Add
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = StandardMultiHeadAttention(embed_dim, n_heads, dropout)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = StandardFFN(embed_dim, hidden_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, K) input
            mask: (N, N) causal mask
            return_attention: If True, also return attention weights

        Returns:
            (B, N, K) output, or tuple with attention weights
        """
        # Attention block (pre-norm)
        residual = x
        x = self.ln1(x)
        if return_attention:
            x, attn_weights = self.attn(x, mask, return_attention=True)
        else:
            x = self.attn(x, mask)
            attn_weights = None
        x = self.dropout(x)
        x = residual + x

        # FFN block (pre-norm)
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        if return_attention:
            return x, attn_weights
        return x


class StandardTransformerLM(nn.Module):
    """
    Standard transformer language model.

    Architecture:
        Embedding → Position Encoding → N × Transformer Blocks → LM Head

    NO gauge theory, NO SO(3), NO KL divergence!
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        vocab_size = config['vocab_size']
        embed_dim = config['embed_dim']
        n_layers = config['n_layers']
        n_heads = config.get('n_heads', 1)
        hidden_dim = config['hidden_dim']
        max_seq_len = config['max_seq_len']
        dropout = config.get('dropout', 0.1)
        tie_embeddings = config.get('tie_embeddings', True)

        # Token embeddings (same as gauge model)
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)

        # Positional embeddings (learned, same as gauge model)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(embed_dim, n_heads, hidden_dim, dropout)
            for _ in range(n_layers)
        ])

        # Output layer
        self.ln_final = nn.LayerNorm(embed_dim)

        if tie_embeddings:
            # Tie input and output embeddings
            self.lm_head = lambda x: F.linear(x, self.token_embed.weight)
        else:
            self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Create causal mask
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0)
        )

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"StandardTransformerLM initialized: {n_params/1e6:.2f}M parameters")

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pad_token_id: int = -100,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (B, N) token indices
            labels: (B, N) target tokens (for loss computation)
            pad_token_id: Token ID for padding (ignored in loss). Default -100.

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        B, N = input_ids.shape
        device = input_ids.device

        # Embed tokens
        x = self.token_embed(input_ids)  # (B, N, K)

        # Add positional embeddings
        pos_ids = torch.arange(N, device=device).unsqueeze(0)  # (1, N)
        x = x + self.pos_embed(pos_ids)

        x = self.dropout(x)

        # Apply transformer blocks
        mask = self.causal_mask[:, :N, :N]
        for block in self.blocks:
            x = block(x, mask)

        # Final layer norm
        x = self.ln_final(x)

        # LM head
        logits = self.lm_head(x)  # (B, N, vocab_size)

        # Compute loss if labels provided
        output = {'logits': logits}

        if labels is not None:
            # NOTE: The dataset already provides pre-shifted targets:
            #   input_ids  = [tok_0, tok_1, ..., tok_{T-1}]
            #   labels     = [tok_1, tok_2, ..., tok_T]
            # So logits[i] (predicting next token after position i) should be
            # compared directly to labels[i] (= tok_{i+1}). No shifting needed.
            # This matches how compute_free_energy_loss handles the gauge model.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='mean',
                ignore_index=pad_token_id,  # Ignore padding tokens in loss
            )
            output['loss'] = loss

        return output

    def forward_with_attention(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass that also returns attention weights for visualization.

        Args:
            input_ids: (B, N) token indices
            targets: Unused, for API compatibility with GaugeTransformerLM

        Returns:
            logits: (B, N, vocab_size) prediction logits
            attn_info: Dict with 'beta' key containing attention weights (B, H, N, N)
        """
        B, N = input_ids.shape
        device = input_ids.device

        # Embed tokens
        x = self.token_embed(input_ids)  # (B, N, K)

        # Add positional embeddings
        pos_ids = torch.arange(N, device=device).unsqueeze(0)  # (1, N)
        x = x + self.pos_embed(pos_ids)

        x = self.dropout(x)

        # Apply transformer blocks, collecting attention weights
        mask = self.causal_mask[:, :N, :N]
        all_attn_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, mask, return_attention=True)
            all_attn_weights.append(attn_weights)

        # Final layer norm
        x = self.ln_final(x)

        # LM head
        logits = self.lm_head(x)  # (B, N, vocab_size)

        # Stack attention weights: (n_layers, B, H, N, N) -> use last layer for viz
        # For compatibility with GaugeTransformerLM, return 'beta' key
        attn_info = {
            'beta': all_attn_weights[-1],  # Last layer attention (B, H, N, N)
            'all_attention': all_attn_weights,  # All layers
        }

        return logits, attn_info

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            prompt_ids: (1, prompt_len) initial tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (optional)
            top_p: Nucleus sampling (optional)

        Returns:
            generated: (1, prompt_len + max_new_tokens) full sequence
        """
        self.eval()
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate if exceeds max_seq_len
            if generated.shape[1] > self.config['max_seq_len']:
                generated = generated[:, -self.config['max_seq_len']:]

            # Forward pass
            output = self.forward(generated)
            logits = output['logits']  # (1, T, V)

            # Get logits for last token
            logits_next = logits[:, -1, :] / temperature  # (1, V)

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits_next, min(top_k, logits_next.size(-1)))
                logits_next[logits_next < v[:, [-1]]] = -float('inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits_next, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits_next[indices_to_remove] = -float('inf')

            # Sample
            probs = F.softmax(logits_next, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {}
        counts['token_embed'] = self.token_embed.weight.numel()
        counts['pos_embed'] = self.pos_embed.weight.numel()

        # Count transformer block parameters
        block_params = 0
        for block in self.blocks:
            block_params += sum(p.numel() for p in block.parameters())
        counts['transformer_blocks'] = block_params

        counts['ln_final'] = sum(p.numel() for p in self.ln_final.parameters())

        if not self.config.get('tie_embeddings', True):
            counts['lm_head'] = self.lm_head.weight.numel()

        counts['total'] = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return counts


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("STANDARD TRANSFORMER BASELINE")
    print("="*70)

    # Match gauge model config
    config = {
        'vocab_size': 256,
        'embed_dim': 11,
        'n_layers': 2,
        'n_heads': 1,  # Single head for K=11
        'hidden_dim': 44,
        'max_seq_len': 32,
        'dropout': 0.1,
        'tie_embeddings': True,
    }

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k:20s}: {v}")

    # Create model
    print("\n" + "="*70)
    model = StandardTransformerLM(config)

    # Count parameters
    print("\n" + "="*70)
    print("PARAMETER BREAKDOWN")
    print("="*70)

    counts = model.count_parameters()
    for name, count in counts.items():
        print(f"  {name:20s}: {count:6d}")

    # Test forward pass
    print("\n" + "="*70)
    print("TEST FORWARD PASS")
    print("="*70)

    B, N = 2, 10
    input_ids = torch.randint(0, config['vocab_size'], (B, N))

    output = model(input_ids, labels=input_ids)

    print(f"  Input shape:  {input_ids.shape}")
    print(f"  Output logits: {output['logits'].shape}")
    print(f"  Loss:         {output['loss'].item():.4f}")

    print("\n✓ Standard transformer baseline ready!")
    print("="*70)