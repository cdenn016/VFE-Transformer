"""
Numerical Sanitization Tracking
================================

Lightweight counter-based tracker for all numerical sanitization events
(sigma clamping, Cholesky fallbacks, KL clamping, NaN replacement, etc.).

Usage:
    from transformer.core.sanitization import san

    # Record events:
    san.record('sigma_clamp', count=3)
    san.record('cholesky_fallback', count=1, value=0.01)  # jitter used

    # Periodic reporting (call from training loop):
    metrics = san.report(step=1000, log_every=100)
    # Returns dict like {'san/sigma_clamp': 47, 'san/cholesky_fallback': 2, ...}
    # and resets counters.
"""

from collections import defaultdict
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SanitizationTracker:
    """Track numerical sanitization events across the model."""

    def __init__(self):
        self.counts: Dict[str, int] = defaultdict(int)
        self.max_values: Dict[str, float] = {}
        self._enabled = True

    def record(self, name: str, count: int = 1, value: Optional[float] = None):
        """Record a sanitization event.

        Args:
            name: Event name (e.g. 'sigma_clamp', 'cholesky_fallback')
            count: Number of elements affected
            value: Optional associated value (e.g. jitter magnitude, clamped value)
        """
        if not self._enabled:
            return
        self.counts[name] += count
        if value is not None:
            prev = self.max_values.get(name)
            if prev is None or value > prev:
                self.max_values[name] = value

    def report(self, step: int, log_every: int = 100) -> Dict[str, float]:
        """Return sanitization metrics and reset counters.

        Only produces output every `log_every` steps. Returns empty dict otherwise.

        Args:
            step: Current training step
            log_every: Report frequency

        Returns:
            Dict of metrics with 'san/' prefix, empty if not a reporting step
        """
        if step % log_every != 0:
            return {}
        if not self.counts:
            return {}

        metrics = {}
        parts = []
        for name, count in sorted(self.counts.items()):
            metrics[f'san/{name}'] = count
            max_val = self.max_values.get(name)
            if max_val is not None:
                metrics[f'san/{name}_max'] = max_val
                parts.append(f'{name}={count}(max={max_val:.2e})')
            else:
                parts.append(f'{name}={count}')

        if parts:
            logger.info(f'[step {step}] sanitization: {", ".join(parts)}')

        # Reset for next interval
        self.counts.clear()
        self.max_values.clear()

        return metrics

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def reset(self):
        self.counts.clear()
        self.max_values.clear()


# Global singleton — import as `from transformer.core.sanitization import san`
san = SanitizationTracker()
