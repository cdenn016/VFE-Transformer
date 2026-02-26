"""
Metrics Tracking for Training
=============================

Unified metrics tracking with CSV logging, extracted from
train_publication.py and train_standard_baseline.py.
"""

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class MetricsTracker:
    """
    Unified metrics tracker with CSV logging.

    Tracks training metrics and writes them to CSV files for
    analysis and visualization.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        experiment_name: str = "experiment",
        include_system_info: bool = True,
    ):
        """
        Initialize metrics tracker.

        Args:
            output_dir: Directory to save metrics (None = don't save)
            experiment_name: Name for the experiment (used in filenames)
            include_system_info: If True, log system info at start
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.experiment_name = experiment_name
        self.history: List[Dict[str, Any]] = []
        self.start_time = time.time()

        # Create output directory
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Log system info
        if include_system_info and self.output_dir:
            self._log_system_info()

    def _log_system_info(self):
        """Log system information to JSON file."""
        import torch

        info = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

        # Try to get git info
        try:
            import subprocess
            info['git_commit'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()[:8]
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            info['git_commit'] = 'unknown'

        # Save to JSON
        info_path = self.output_dir / f"{self.experiment_name}_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

    def log(self, metrics: Dict[str, Any], step: int):
        """
        Log metrics for a training step.

        Args:
            metrics: Dict of metric names to values
            step: Current training step
        """
        record = {
            'step': step,
            'time': time.time() - self.start_time,
            **metrics
        }
        self.history.append(record)

    def log_eval(self, metrics: Dict[str, Any], step: int):
        """
        Log evaluation metrics.

        Args:
            metrics: Dict of metric names to values
            step: Current training step
        """
        record = {
            'step': step,
            'time': time.time() - self.start_time,
            'is_eval': True,
            **metrics
        }
        self.history.append(record)

    def save(self, suffix: str = ""):
        """
        Save metrics to CSV file.

        Args:
            suffix: Optional suffix for filename
        """
        if not self.output_dir or not self.history:
            return

        filename = f"{self.experiment_name}_metrics{suffix}.csv"
        filepath = self.output_dir / filename

        # Collect all unique keys
        all_keys = set()
        for record in self.history:
            all_keys.update(record.keys())

        # Sort keys for consistent ordering
        fieldnames = ['step', 'time'] + sorted(k for k in all_keys if k not in ['step', 'time'])

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.history)

        print(f"Saved metrics to {filepath}")

    def get_best(self, metric: str, mode: str = 'min') -> Optional[Dict[str, Any]]:
        """
        Get the record with best value for a metric.

        Args:
            metric: Name of the metric
            mode: 'min' or 'max'

        Returns:
            Record dict with best value, or None if metric not found
        """
        records_with_metric = [r for r in self.history if metric in r]
        if not records_with_metric:
            return None

        if mode == 'min':
            return min(records_with_metric, key=lambda r: r[metric])
        else:
            return max(records_with_metric, key=lambda r: r[metric])

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get the most recent record."""
        return self.history[-1] if self.history else None

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training run.

        Returns:
            Dict with summary statistics
        """
        if not self.history:
            return {}

        summary = {
            'total_steps': self.history[-1]['step'],
            'total_time_seconds': self.history[-1]['time'],
            'total_time_hours': self.history[-1]['time'] / 3600,
        }

        # Find best metrics
        for metric in ['loss', 'val_loss', 'perplexity', 'val_perplexity']:
            best = self.get_best(metric, mode='min')
            if best:
                summary[f'best_{metric}'] = best[metric]
                summary[f'best_{metric}_step'] = best['step']

        return summary


def format_metrics(metrics: Dict[str, Any], precision: int = 4) -> str:
    """
    Format metrics dict for printing.

    Args:
        metrics: Dict of metric names to values
        precision: Decimal precision for floats

    Returns:
        Formatted string
    """
    parts = []
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            parts.append(f"{key}={value:.{precision}f}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)
