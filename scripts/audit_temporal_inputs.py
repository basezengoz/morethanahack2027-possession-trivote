"""Audit target inputs and head outputs without reading any reference labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from possession.io import load_tracking
from possession.tagging import load_tagging
from possession.train import load_checkpoint, zero_tags
from possession.model import flatten_features, infer_evidence
from possession.quality import clean_tracking
from possession.features import build_features
from possession.teacher import build_evidence
from possession.predict import load_config, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',default='data'); p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True); a=p.parse_args()
    out=Path(a.output).resolve()
    if not out.is_relative_to(Path('data/runs').resolve()): raise ValueError('private output required')
    cfg=load_config('configs/rules.yaml'); tracking=load_tracking(a.data_dir)
    tags=load_tagging(a.data_dir,len(tracking.frame),tracking.fps,cfg.get('tagging',{}))
    f=build_features(clean_tracking(tracking,cfg.get('quality',{}),tags.restart),cfg.get('features',{}))
    base=build_evidence(f,tags,cfg); model,norm,ckpt=load_checkpoint(a.checkpoint,'cpu')
    ztags=zero_tags(len(tracking.frame)); raw=flatten_features(f,ztags)
    if getattr(model,'feature_version',1)==2:
        from possession.model_v2 import enrich
        raw=enrich(raw,f.blocks,f.fps)
    normalized=norm.transform(raw)[:,:raw.shape[1]]
    inside=np.zeros(len(raw),bool)
    for start,end in f.blocks: inside[start:end]=True
    columns={}; offset=0
    for name,width in ckpt['schema']:
        values=raw[inside,offset:offset+width]; zs=np.abs(normalized[inside,offset:offset+width])
        valid=np.isfinite(values)
        columns[name]={'raw_p50_p99_max':np.quantile(values[valid],[.5,.99,1.]).tolist() if valid.any() else None,
                       'abs_z_p50_p99_max':np.quantile(zs[valid],[.5,.99,1.]).tolist() if valid.any() else None,
                       'fraction_abs_z_over_10':float((zs[valid]>10).mean()) if valid.any() else None,
                       'train_mean':norm.mean[offset:offset+width].tolist(),
                       'train_scale':norm.scale[offset:offset+width].tolist()}
        offset+=width
    e=infer_evidence(model,f,ztags,base,norm,dead_head_trained=ckpt['stats']['dead_head_trained'])
    groups={}
    for name,mask in (('available',inside & (f.values['ball_available']>0)),
                      ('missing',inside & ~(f.values['ball_available']>0))):
        scores=e.state_scores[mask]
        groups[name]={'frames':int(mask.sum()),'mean_scores_HAD':scores.mean(axis=0).tolist() if len(scores) else None,
                      'fraction_max_score_over_099':float((scores.max(axis=1)>.99).mean()) if len(scores) else None}
    report={'checkpoint_sha256':sha256(a.checkpoint),'script_sha256':sha256(__file__),
            'scope':'target tracking only; no target GT; official processing blocks',
            'features':columns,'head_outputs':groups}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'head_outputs':groups,'most_extreme':sorted(columns,key=lambda k:(columns[k]['abs_z_p50_p99_max'] or [0,0,0])[-1],reverse=True)[:8]}))


if __name__=='__main__': main()
