"""Experimental centered TCN; no dependency on target-match ground truth."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .contracts import Evidence

# Each scalar has a numeric value and a separate observed mask. Team pairs are
# adjacent, enabling exact H/A swap on both values and masks.
SCHEMA = (
    ('nearest_distance', 2), ('control_support', 2), ('team_speed', 2),
    ('team_spread', 2), ('nearby_player_count', 2), ('ball_speed', 1),
    ('boundary_distance', 1), ('ball_observed', 1), ('ball_interpolated', 1),
    ('ball_rejected', 1), ('ball_available', 1), ('tag_team_support', 2),
    ('tag_restart', 1), ('tag_conflict', 1),
)
N_RAW = sum(width for _, width in SCHEMA)
TEAM_PAIRS = tuple(sum(w for _, w in SCHEMA[:i]) for i, (name, w) in enumerate(SCHEMA)
                   if w == 2)
TAG_START = sum(w for name, w in SCHEMA if not name.startswith('tag_'))


def flatten_features(features, tags):
    """Return (T, N_RAW) in SCHEMA order; NaN records an unknown value."""
    n = len(features.values['ball_available'])
    columns = []
    for name, width in SCHEMA:
        a = np.asarray(getattr(tags, name[4:]) if name.startswith('tag_') else features.values[name], dtype=np.float32)
        if width == 1 and a.shape == (n,):
            a = a[:, None]
        if a.shape != (n, width):
            raise ValueError(f'{name}: expected {(n, width)}, got {a.shape}')
        columns.append(a)
    return np.concatenate(columns, axis=1)


@dataclass
class Normalizer:
    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=np.float32)
        self.scale = np.asarray(self.scale, dtype=np.float32)
        if (self.mean.shape != (N_RAW,) or self.scale.shape != (N_RAW,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or np.any(self.scale <= 0)):
            raise ValueError('invalid normalization: finite means and positive finite scales required')

    @classmethod
    def fit(cls, arrays):
        """Fit only on training matches, using finite observations per channel."""
        if not arrays:
            raise ValueError('training arrays required')
        data = np.concatenate(arrays, axis=0).astype(np.float64)
        if data.ndim != 2 or data.shape[1] != N_RAW or not len(data):
            raise ValueError('feature schema mismatch')
        valid = np.isfinite(data)
        count = valid.sum(axis=0)
        mean = np.divide(np.where(valid, data, 0).sum(axis=0), count, out=np.zeros(N_RAW), where=count > 0)
        sq = np.where(valid, data - mean, 0) ** 2
        var = np.divide(sq.sum(axis=0), count, out=np.ones(N_RAW), where=count > 0)
        # A constant/unseen training channel must not amplify a new input by 1e6.
        scale = np.where(var > 1e-12, np.sqrt(var), 1.)
        # Preserve team-swap symmetry of normalization.
        for i in TEAM_PAIRS:
            pooled_count = count[i] + count[i+1]
            if pooled_count:
                pooled_mean = (mean[i]*count[i] + mean[i+1]*count[i+1])/pooled_count
                pooled_var = ((var[i]+(mean[i]-pooled_mean)**2)*count[i] + (var[i+1]+(mean[i+1]-pooled_mean)**2)*count[i+1])/pooled_count
                mean[i:i+2] = pooled_mean
                scale[i:i+2] = np.sqrt(pooled_var) if pooled_var > 1e-12 else 1.
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != N_RAW:
            raise ValueError('feature schema mismatch')
        mask = np.isfinite(raw)
        value = np.where(mask, (raw - self.mean) / self.scale, 0)
        return np.concatenate((value, mask.astype(np.float32)), axis=1).astype(np.float32)


def swap_teams(raw, labels=None):
    output = np.array(raw, copy=True)
    for i in TEAM_PAIRS:
        output[:, [i, i+1]] = output[:, [i+1, i]]
    if labels is None:
        return output
    target = np.array(labels, copy=True)
    target[labels == 0], target[labels == 1] = 1, 0
    return output, target


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.activation = nn.GELU()

    def forward(self, x):
        return x + self.activation(self.conv(x))


class TemporalModel(nn.Module):
    """8 centered convolutions give a 511-frame receptive field (radius 255)."""
    receptive_radius = 255

    def __init__(self, channels=128):
        super().__init__()
        self.input = nn.Conv1d(2 * N_RAW, channels, 1)
        self.blocks = nn.Sequential(*(ResidualBlock(channels, 2**i) for i in range(8)))
        self.dead_head = nn.Conv1d(channels, 1, 1)
        self.team_head = nn.Conv1d(channels, 2, 1)

    def forward(self, x):
        h = self.blocks(self.input(x))
        return self.dead_head(h).squeeze(1), self.team_head(h)


def infer_evidence(model, features, tags, baseline_evidence, normalizer, device='cpu', dead_head_trained=False, center_frames=512, context_frames=256, control_gate=False):
    """Chunk per half and preserve teacher transitions; optionally gate H/A by control.

    control_gate is an explicit integration option, not a claim of validation gain.
    It blends the conditional H/A head with rules in proportion to observed control,
    retaining the exact rule qD when the D head is untrained. The default preserves
    the original research checkpoint's inference policy.
    """
    if center_frames <= 0 or context_frames < model.receptive_radius:
        raise ValueError('inference context is shorter than receptive field')
    raw = flatten_features(features, tags)
    if getattr(model, 'feature_version', 1) == 2:
        from .model_v2 import enrich
        raw = enrich(raw, features.blocks, features.fps)
    x = normalizer.transform(raw)
    base = np.asarray(baseline_evidence.state_scores, dtype=np.float32)
    if base.shape != (len(x), 3):
        raise ValueError('baseline score shape mismatch')
    if (not np.isfinite(base).all() or np.any(base < 0)
            or not np.allclose(base.sum(axis=1), 1, atol=1e-5)):
        raise ValueError('baseline scores must be finite normalized H/A/D weights')
    weight = np.ones(len(x), np.float32)
    if control_gate:
        available = np.asarray(features.values['ball_available'], dtype=bool)
        control = np.asarray(features.values['control_support'], dtype=np.float32)
        if available.shape != (len(x),) or control.shape != (len(x), 2):
            raise ValueError('control gating feature shape mismatch')
        weight = available * np.clip(np.nan_to_num(control, nan=0).max(axis=1), 0, 1)
    conditional_base = np.divide(base[:, :2], (1-base[:, 2:3]),
                                 out=np.full((len(x), 2), .5, np.float32), where=(1-base[:, 2:3]) > 0)
    scores = base.copy()
    model = model.to(device).eval()
    with torch.inference_mode():
        for start, end in features.blocks:
            for left in range(start, end, center_frames):
                right = min(left + center_frames, end)
                lo, hi = max(start, left-context_frames), min(end, right+context_frames)
                sample = torch.from_numpy(x[lo:hi].T[None]).to(device)
                dead, team = model(sample)
                if not bool(torch.isfinite(team).all()) or (dead_head_trained and not bool(torch.isfinite(dead).all())):
                    raise ValueError('TCN produced nonfinite logits')
                conditional = torch.softmax(team[0, :, left-lo:right-lo], dim=0).T.cpu().numpy()
                blend = weight[left:right, None]
                conditional = blend * conditional + (1-blend) * conditional_base[left:right]
                qd = torch.sigmoid(dead[0, left-lo:right-lo]).cpu().numpy() if dead_head_trained else base[left:right, 2]
                scores[left:right, :2] = conditional * (1-qd[:, None])
                scores[left:right, 2] = qd
                if control_gate and not dead_head_trained:
                    # Preserve exact weights (including rounding) where no control exists.
                    neutral = weight[left:right] == 0
                    scores[left:right][neutral] = base[left:right][neutral]
    return Evidence(scores, baseline_evidence.change.copy(), baseline_evidence.stop.copy(), baseline_evidence.restart.copy())
