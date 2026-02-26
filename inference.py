#!/usr/bin/env python3
"""
Model Inference & Qualitative Analysis
=======================================

Load a trained GaugeTransformerLM and generate text, visualize attention,
and produce qualitative examples for analysis.

Usage:
    # Interactive mode
    python inference.py --checkpoint path/to/best_model.pt

    # Single prompt
    python inference.py --checkpoint path/to/best_model.pt --prompt "The quick brown"

    # Generate multiple samples
    python inference.py --checkpoint path/to/best_model.pt --prompt "AI is" --num_samples 5

    # Visualize attention
    python inference.py --checkpoint path/to/best_model.pt --prompt "The cat sat" --visualize

Author: Claude
Date: January 2026
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from transformer.utils.checkpoint import load_model, get_tokenizer


class GaugeTransformerInference:
    """
    Inference wrapper for GaugeTransformerLM.

    Provides:
    - Text generation with various sampling strategies
    - Attention pattern extraction
    - Belief state inspection
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        dataset_name: str = 'wikitext-103',
    ):
        """
        Initialize inference engine.

        Args:
            checkpoint_path: Path to model checkpoint (best_model.pt)
            device: Device to use ('cuda', 'cpu', or None for auto)
            dataset_name: Dataset for tokenizer ('wikitext-103' or 'wikitext-2')
        """
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model, self.config = load_model(checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        print(f"Loading tokenizer ({dataset_name})...")
        self.tokenizer = get_tokenizer(self.config, dataset_name=dataset_name)

        if self.tokenizer is None:
            raise RuntimeError("Could not load tokenizer. Install tiktoken or transformers.")

        print(f"Model loaded: {self.config['embed_dim']}D embeddings, "
              f"{self.config['n_layers']} layers, "
              f"vocab size {self.config['vocab_size']}")
        print(f"Device: {self.device}")

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        token_ids = self.tokenizer.encode(text)
        return torch.tensor([token_ids], device=self.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        ids = token_ids.squeeze().tolist()
        if isinstance(ids, int):
            ids = [ids]
        return self.tokenizer.decode(ids)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        num_samples: int = 1,
    ) -> List[str]:
        """
        Generate text continuations.

        Args:
            prompt: Input text to continue
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering (None to disable)
            top_p: Nucleus sampling threshold (None to disable)
            num_samples: Number of independent samples to generate

        Returns:
            List of generated text strings (including prompt)
        """
        prompt_ids = self.encode(prompt)
        results = []

        for _ in range(num_samples):
            with torch.no_grad():
                generated_ids = self.model.generate(
                    prompt_ids=prompt_ids.clone(),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

            text = self.decode(generated_ids[0])
            results.append(text)

        return results

    def get_next_token_probs(
        self,
        text: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Get probability distribution over next token.

        Args:
            text: Input text
            top_k: Number of top tokens to return

        Returns:
            List of (token_string, probability) tuples
        """
        token_ids = self.encode(text)

        with torch.no_grad():
            logits = self.model(token_ids)  # (1, seq_len, vocab_size)
            last_logits = logits[0, -1, :]  # (vocab_size,)
            probs = F.softmax(last_logits, dim=-1)

        # Get top-k
        top_probs, top_ids = torch.topk(probs, min(top_k, probs.size(0)))

        results = []
        for prob, tok_id in zip(top_probs.tolist(), top_ids.tolist()):
            try:
                token_str = self.decode(torch.tensor([tok_id]))
            except (KeyError, IndexError, RuntimeError):
                token_str = f"[{tok_id}]"
            results.append((token_str, prob))

        return results

    def get_attention_patterns(
        self,
        text: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention patterns for visualization.

        Args:
            text: Input text

        Returns:
            Dict with:
                - 'beta': (n_heads, seq_len, seq_len) attention weights
                - 'kl': (n_heads, seq_len, seq_len) KL divergences
                - 'tokens': List of token strings
        """
        token_ids = self.encode(text)

        with torch.no_grad():
            logits, attn_info = self.model.forward_with_attention(
                token_ids,
                targets=None,
            )

        # Decode tokens for labeling
        tokens = []
        for i in range(token_ids.shape[1]):
            try:
                tok_str = self.decode(token_ids[0, i:i+1])
                # Truncate long tokens for display
                if len(tok_str) > 10:
                    tok_str = tok_str[:8] + ".."
            except (KeyError, IndexError, RuntimeError):
                tok_str = f"[{token_ids[0, i].item()}]"
            tokens.append(tok_str)

        return {
            'beta': attn_info['beta'][0].cpu(),  # (n_heads, N, N)
            'kl': attn_info['kl'][0].cpu(),      # (n_heads, N, N)
            'mu': attn_info['mu'][0].cpu(),      # (N, K)
            'tokens': tokens,
        }

    def get_belief_states(
        self,
        text: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract belief states (mu, sigma, phi) for analysis.

        Args:
            text: Input text

        Returns:
            Dict with mu, sigma, phi tensors and token strings
        """
        token_ids = self.encode(text)

        with torch.no_grad():
            logits, agent_states = self.model(token_ids, return_agents=True)

        # Decode tokens
        tokens = []
        for i in range(token_ids.shape[1]):
            try:
                tok_str = self.decode(token_ids[0, i:i+1])
            except (KeyError, IndexError, RuntimeError):
                tok_str = f"[{token_ids[0, i].item()}]"
            tokens.append(tok_str)

        return {
            'mu': agent_states['mu'][0].cpu(),      # (N, K)
            'sigma': agent_states['sigma'][0].cpu() if agent_states['sigma'] is not None else None,
            'phi': agent_states['phi'][0].cpu(),    # (N, phi_dim)
            'tokens': tokens,
            'logits': logits[0].cpu(),              # (N, V)
        }


def visualize_attention(
    attn_data: Dict,
    save_path: Optional[str] = None,
    head_idx: int = 0,
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Visualize attention patterns.

    Args:
        attn_data: Output from get_attention_patterns()
        save_path: Path to save figure (None for display)
        head_idx: Which attention head to visualize
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    beta = attn_data['beta']
    kl = attn_data['kl']
    tokens = attn_data['tokens']
    n_heads = beta.shape[0]

    # Create figure with attention and KL subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Attention weights
    ax1 = axes[0]
    im1 = ax1.imshow(
        beta[head_idx].numpy(),
        cmap='Blues',
        aspect='auto',
    )
    ax1.set_title(f'Attention Weights (Head {head_idx})')
    ax1.set_xlabel('Key Position (j)')
    ax1.set_ylabel('Query Position (i)')
    ax1.set_xticks(range(len(tokens)))
    ax1.set_yticks(range(len(tokens)))
    ax1.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(tokens, fontsize=8)
    plt.colorbar(im1, ax=ax1, label='β_ij')

    # KL divergences
    ax2 = axes[1]
    # Use log scale for KL (can have large values)
    kl_plot = kl[head_idx].numpy()
    kl_plot = np.clip(kl_plot, 1e-6, None)  # Avoid log(0)
    im2 = ax2.imshow(
        np.log10(kl_plot + 1),
        cmap='Reds',
        aspect='auto',
    )
    ax2.set_title(f'KL Divergences (Head {head_idx})')
    ax2.set_xlabel('Key Position (j)')
    ax2.set_ylabel('Query Position (i)')
    ax2.set_xticks(range(len(tokens)))
    ax2.set_yticks(range(len(tokens)))
    ax2.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax2.set_yticklabels(tokens, fontsize=8)
    plt.colorbar(im2, ax=ax2, label='log₁₀(KL + 1)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_beliefs(
    belief_data: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
):
    """
    Visualize belief state evolution across sequence.

    Args:
        belief_data: Output from get_belief_states()
        save_path: Path to save figure
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available.")
        return

    mu = belief_data['mu']  # (N, K)
    tokens = belief_data['tokens']

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Belief means heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(mu.numpy().T, aspect='auto', cmap='RdBu_r')
    ax1.set_title('Belief Means (μ)')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Embedding Dimension')
    ax1.set_xticks(range(len(tokens)))
    ax1.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    plt.colorbar(im1, ax=ax1)

    # Belief norms
    ax2 = axes[1]
    norms = torch.norm(mu, dim=-1).numpy()
    ax2.bar(range(len(tokens)), norms)
    ax2.set_title('Belief Norms ||μ||')
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('L2 Norm')
    ax2.set_xticks(range(len(tokens)))
    ax2.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved belief visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def print_generation_samples(
    inference: GaugeTransformerInference,
    prompts: List[str],
    num_samples: int = 3,
    max_tokens: int = 50,
    temperature: float = 0.8,
):
    """
    Print formatted generation samples for qualitative analysis.

    Args:
        inference: Inference engine
        prompts: List of prompts to try
        num_samples: Samples per prompt
        max_tokens: Max tokens per sample
        temperature: Sampling temperature
    """
    print("\n" + "=" * 70)
    print("QUALITATIVE GENERATION SAMPLES")
    print("=" * 70)
    print(f"Temperature: {temperature}, Max tokens: {max_tokens}")
    print("=" * 70)

    for prompt in prompts:
        print(f"\n{'─' * 60}")
        print(f"PROMPT: \"{prompt}\"")
        print(f"{'─' * 60}")

        # Show next-token distribution
        top_tokens = inference.get_next_token_probs(prompt, top_k=5)
        print("\nNext token probabilities:")
        for tok, prob in top_tokens:
            bar = "█" * int(prob * 30)
            print(f"  {prob:5.1%} {bar} '{tok}'")

        # Generate samples
        print(f"\nGenerated continuations ({num_samples} samples):")
        samples = inference.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            num_samples=num_samples,
        )

        for i, sample in enumerate(samples, 1):
            # Highlight the generated part
            prompt_end = len(prompt)
            generated = sample[prompt_end:] if len(sample) > prompt_end else ""
            print(f"\n  [{i}] {prompt}\033[92m{generated}\033[0m")

    print("\n" + "=" * 70)


def interactive_mode(inference: GaugeTransformerInference):
    """
    Interactive text generation mode.

    Args:
        inference: Inference engine
    """
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Commands:")
    print("  /quit       - Exit")
    print("  /temp N     - Set temperature (e.g., /temp 0.5)")
    print("  /tokens N   - Set max tokens (e.g., /tokens 100)")
    print("  /samples N  - Set num samples (e.g., /samples 5)")
    print("  /attention  - Show attention for last input")
    print("  /probs      - Show next-token probabilities")
    print("=" * 70)

    temperature = 0.8
    max_tokens = 50
    num_samples = 1
    last_input = None

    while True:
        try:
            user_input = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith('/'):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == '/quit':
                print("Goodbye!")
                break
            elif cmd == '/temp' and len(parts) > 1:
                try:
                    temperature = float(parts[1])
                    print(f"Temperature set to {temperature}")
                except ValueError:
                    print("Invalid temperature")
            elif cmd == '/tokens' and len(parts) > 1:
                try:
                    max_tokens = int(parts[1])
                    print(f"Max tokens set to {max_tokens}")
                except ValueError:
                    print("Invalid token count")
            elif cmd == '/samples' and len(parts) > 1:
                try:
                    num_samples = int(parts[1])
                    print(f"Num samples set to {num_samples}")
                except ValueError:
                    print("Invalid sample count")
            elif cmd == '/attention' and last_input:
                print("\nExtracting attention patterns...")
                attn_data = inference.get_attention_patterns(last_input)
                visualize_attention(attn_data)
            elif cmd == '/probs' and last_input:
                probs = inference.get_next_token_probs(last_input, top_k=10)
                print("\nNext token probabilities:")
                for tok, prob in probs:
                    bar = "█" * int(prob * 40)
                    print(f"  {prob:5.1%} {bar} '{tok}'")
            else:
                print("Unknown command or missing argument")
            continue

        # Generate text
        last_input = user_input
        print(f"\nGenerating ({num_samples} sample(s), temp={temperature})...")

        samples = inference.generate(
            user_input,
            max_new_tokens=max_tokens,
            temperature=temperature,
            num_samples=num_samples,
        )

        for i, sample in enumerate(samples, 1):
            prompt_end = len(user_input)
            generated = sample[prompt_end:] if len(sample) > prompt_end else ""
            if num_samples > 1:
                print(f"\n[{i}] {user_input}\033[92m{generated}\033[0m")
            else:
                print(f"\n{user_input}\033[92m{generated}\033[0m")


def main():
    parser = argparse.ArgumentParser(
        description='Gauge Transformer Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python inference.py --checkpoint experiments/best_model.pt

  # Single prompt
  python inference.py --checkpoint model.pt --prompt "Once upon a time"

  # Multiple samples
  python inference.py --checkpoint model.pt --prompt "AI is" --num_samples 5

  # Visualize attention
  python inference.py --checkpoint model.pt --prompt "The cat sat" --visualize

  # Qualitative examples for paper
  python inference.py --checkpoint model.pt --examples
        """
    )

    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint (best_model.pt)',
    )
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        default=None,
        help='Text prompt for generation',
    )
    parser.add_argument(
        '--max_tokens', '-t',
        type=int,
        default=50,
        help='Maximum tokens to generate (default: 50)',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8)',
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=40,
        help='Top-k sampling (default: 40, 0 to disable)',
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Nucleus sampling p (default: 0.9, 0 to disable)',
    )
    parser.add_argument(
        '--num_samples', '-n',
        type=int,
        default=1,
        help='Number of samples to generate (default: 1)',
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Visualize attention patterns',
    )
    parser.add_argument(
        '--save_attention',
        type=str,
        default=None,
        help='Path to save attention visualization',
    )
    parser.add_argument(
        '--examples', '-e',
        action='store_true',
        help='Generate qualitative examples with standard prompts',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='wikitext-103',
        choices=['wikitext-103', 'wikitext-2'],
        help='Dataset for tokenizer (default: wikitext-103)',
    )

    args = parser.parse_args()

    # Validate checkpoint
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Initialize inference engine
    inference = GaugeTransformerInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        dataset_name=args.dataset,
    )

    # Handle top_k/top_p=0 as disabled
    top_k = args.top_k if args.top_k > 0 else None
    top_p = args.top_p if args.top_p > 0 else None

    # Qualitative examples mode
    if args.examples:
        example_prompts = [
            "The",
            "In the beginning",
            "Scientists have discovered",
            "The meaning of life is",
            "Once upon a time, there was",
        ]
        print_generation_samples(
            inference,
            example_prompts,
            num_samples=3,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        return

    # Single prompt mode
    if args.prompt:
        # Generate
        samples = inference.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=top_k,
            top_p=top_p,
            num_samples=args.num_samples,
        )

        print("\n" + "=" * 60)
        print(f"Prompt: \"{args.prompt}\"")
        print("=" * 60)

        for i, sample in enumerate(samples, 1):
            prompt_end = len(args.prompt)
            generated = sample[prompt_end:] if len(sample) > prompt_end else ""
            if args.num_samples > 1:
                print(f"\n[{i}] {args.prompt}\033[92m{generated}\033[0m")
            else:
                print(f"\n{args.prompt}\033[92m{generated}\033[0m")

        # Visualize attention if requested
        if args.visualize or args.save_attention:
            print("\nExtracting attention patterns...")
            attn_data = inference.get_attention_patterns(args.prompt)
            visualize_attention(
                attn_data,
                save_path=args.save_attention,
            )

        return

    # Interactive mode (default)
    interactive_mode(inference)


if __name__ == '__main__':
    main()
