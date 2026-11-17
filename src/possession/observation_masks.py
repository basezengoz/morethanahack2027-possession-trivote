"""Transfer observation availability patterns without transferring state labels."""
from dataclasses import replace
import numpy as np


def transfer_ball_mask(tracking, pattern, offset=0):
    """Cycle source-half masks within each destination half; preserve raw observations."""
    source_blocks=pattern.blocks or [(0,len(pattern.frame))]
    if not source_blocks: raise ValueError('nonempty pattern blocks required')
    drop=np.zeros(len(tracking.frame),bool)
    for i,(start,end) in enumerate(tracking.blocks or [(0,len(tracking.frame))]):
        a,b=source_blocks[i % len(source_blocks)]
        if b<=a: raise ValueError('empty pattern block')
        missing=~np.asarray(pattern.ball_observed[a:b],bool)
        drop[start:end]=missing[(np.arange(end-start)+int(offset)) % len(missing)]
    drop &= np.asarray(tracking.ball_observed,bool)
    ball=tracking.ball_xy_raw.copy(); ball[drop]=np.nan
    observed=tracking.ball_observed.copy(); observed[drop]=False
    return replace(tracking,ball_xy_raw=ball,ball_observed=observed), drop
