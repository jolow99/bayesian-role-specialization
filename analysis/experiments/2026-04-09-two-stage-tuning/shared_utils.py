"""Checkpoint & resume helpers for grid search experiments."""

import json
import os
import tempfile
import numpy as np


def load_checkpoint(path):
    """Load checkpoint results. Returns [] if file doesn't exist."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save_checkpoint(path, results):
    """Atomic write via tmp file + os.rename()."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(results, f, indent=2, default=_json_default)
        os.rename(tmp, path)
    except:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def get_completed_keys(results, key_fields):
    """Get set of parameter combo tuples already evaluated."""
    return {tuple(r[k] for k in key_fields) for r in results}


def pick_best(results, metric='combo_r'):
    """Result with highest metric value (skipping NaN)."""
    valid = [r for r in results if not np.isnan(r.get(metric, float('nan')))]
    if not valid:
        return results[0] if results else None
    return max(valid, key=lambda r: r[metric])


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
