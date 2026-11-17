"""Versioned research features and auxiliary boundary head for the TCN."""
import numpy as np
import torch
from torch import nn
from .model import N_RAW, SCHEMA, Normalizer, ResidualBlock

EXTRAS = ('distance_advantage_m', 'control_advantage', 'ball_speed_change_mps',
          'missing_ball_seconds', 'since_observed_seconds')
SCHEMA_V2 = SCHEMA + tuple((name, 1) for name in EXTRAS)
WIDTH = N_RAW + len(EXTRAS)


def enrich(raw, blocks, fps=25., previous_ages=None):
    """Append block-local causal context; no reference labels enter features."""
    raw = np.asarray(raw, np.float32)
    if raw.ndim != 2 or raw.shape[1] != N_RAW: raise ValueError('v2 base schema mismatch')
    offsets = {}; offset = 0
    for name, width in SCHEMA:
        offsets[name] = offset; offset += width
    out = np.full((len(raw), len(EXTRAS)), np.nan, np.float32)
    d = offsets['nearest_distance']; c = offsets['control_support']; s = offsets['ball_speed']
    out[:,0] = raw[:,d] - raw[:,d+1]
    out[:,1] = raw[:,c] - raw[:,c+1]
    available = raw[:,offsets['ball_available']] > 0.5
    observed = raw[:,offsets['ball_observed']] > 0.5
    for start, end in blocks:
        out[start,2] = 0 if np.isfinite(raw[start,s]) else np.nan
        out[start+1:end,2] = raw[start+1:end,s] - raw[start:end-1,s]
        missing_age, observed_age = previous_ages if previous_ages is not None and start == 0 else (0., np.nan)
        for i in range(start,end):
            missing_age = 0. if available[i] else missing_age + 1/fps
            observed_age = 0. if observed[i] else observed_age + 1/fps
            out[i,3] = missing_age
            out[i,4] = observed_age
    return np.concatenate((raw,out),axis=1)


class NormalizerV2:
    def __init__(self, mean, scale):
        self.mean=np.asarray(mean,np.float32); self.scale=np.asarray(scale,np.float32)
        if self.mean.shape != (WIDTH,) or self.scale.shape != (WIDTH,) or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError('invalid v2 normalizer')
    @classmethod
    def fit(cls, arrays):
        base=Normalizer.fit([a[:,:N_RAW] for a in arrays])
        x=np.concatenate([a[:,N_RAW:] for a in arrays]).astype(np.float64)
        finite=np.isfinite(x); count=finite.sum(axis=0)
        mean=np.divide(np.where(finite,x,0).sum(axis=0),count,out=np.zeros(len(EXTRAS)),where=count>0)
        # Team advantage features change sign when teams swap.
        mean[:2]=0
        var=np.divide(np.where(finite,x-mean,0).__pow__(2).sum(axis=0),count,out=np.ones(len(EXTRAS)),where=count>0)
        scale=np.where(var>1e-12,np.sqrt(var),1.)
        return cls(np.r_[base.mean,mean],np.r_[base.scale,scale])
    def transform(self, raw):
        raw=np.asarray(raw,np.float32)
        if raw.ndim != 2 or raw.shape[1] != WIDTH: raise ValueError('v2 feature schema mismatch')
        mask=np.isfinite(raw); value=np.where(mask,(raw-self.mean)/self.scale,0)
        return np.concatenate((value,mask.astype(np.float32)),axis=1).astype(np.float32)


class TemporalModelV2(nn.Module):
    receptive_radius=255
    feature_version=2
    def __init__(self,channels=128):
        super().__init__()
        self.input=nn.Conv1d(2*WIDTH,channels,1)
        self.blocks=nn.Sequential(*(ResidualBlock(channels,2**i) for i in range(8)))
        self.dead_head=nn.Conv1d(channels,1,1); self.team_head=nn.Conv1d(channels,2,1)
        self.boundary_head=nn.Conv1d(channels,1,1)
        self.boundary_logits=None
    def forward(self,x):
        h=self.blocks(self.input(x))
        self.boundary_logits=self.boundary_head(h).squeeze(1)
        return self.dead_head(h).squeeze(1),self.team_head(h)
