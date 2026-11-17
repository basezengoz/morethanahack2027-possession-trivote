"""Measure reference possession recall by duration on a separate DFL match."""
import argparse
import json
from pathlib import Path
import numpy as np
from possession.dfl_data import load_dfl
from possession.train import load_checkpoint, zero_tags
from possession.predict import load_config, sha256
from possession.quality import clean_tracking
from possession.features import build_features
from possession.teacher import build_evidence
from possession.model import infer_evidence
from possession.decode import decode_labels
from possession.external_evaluation import _runs


def duration_recall(labels, pred, mask, blocks, fps=25, threshold=.5):
    """Greedy same-team IoU matching, excluding spells touching coverage edges."""
    truth=[]; predicted=[]
    for start,end in blocks:
        for a,b in _runs(mask[start:end]):
            lo,hi=start+a,start+b
            p=pred[lo:hi].copy()
            known=np.flatnonzero(p!='X')
            if len(known):
                carry=np.maximum.accumulate(np.where(p!='X',np.arange(len(p)),-1))
                carry[carry<0]=known[0]; p=p[carry]
            for seq, dest in ((labels[lo:hi],truth),(p,predicted)):
                for code in (0,1):
                    value=code if dest is truth else 'HA'[code]
                    for left,right in _runs(seq == value):
                        if left>0 and right<hi-lo: dest.append((lo+left,lo+right,code))
    pairs=sorted(((min(b,d)-max(a,c))/(max(b,d)-min(a,c)),i,j)
                 for i,(a,b,k) in enumerate(truth) for j,(c,d,l) in enumerate(predicted)
                 if k==l and min(b,d)>max(a,c))
    used_t=set(); used_p=set()
    for iou,i,j in reversed(pairs):
        if iou >= threshold and i not in used_t and j not in used_p:
            used_t.add(i); used_p.add(j)
    bins={}
    for name,low,high in (('under_2s',0,2),('2_to_5s',2,5),('5_to_10s',5,10),('10s_or_more',10,float('inf'))):
        ids=[i for i,(a,b,_) in enumerate(truth) if low <= (b-a)/fps < high]
        matches=sum(i in used_t for i in ids)
        bins[name]={'support':len(ids),'matched':matches,'recall':matches/len(ids) if ids else None}
    return {'iou_threshold':threshold,'reference_spells':len(truth),'predicted_spells':len(predicted),'duration_bins':bins}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('positions','info','output'): p.add_argument('--'+key,required=True)
    p.add_argument('--checkpoint',action='append',required=True); a=p.parse_args()
    out=Path(a.output).resolve()
    if not out.is_relative_to(Path('data/runs').resolve()): raise ValueError('private output required')
    m=load_dfl(a.positions,a.info); cfg=load_config('configs/rules.yaml'); tags=zero_tags(len(m.labels))
    f=build_features(clean_tracking(m.tracking,cfg.get('quality',{}),tags.restart),cfg.get('features',{}))
    base=build_evidence(f,tags,cfg); results={}
    for path in a.checkpoint:
        model,norm,ckpt=load_checkpoint(path,'cpu')
        e=infer_evidence(model,f,tags,base,norm,dead_head_trained=ckpt['stats']['dead_head_trained'])
        pred=decode_labels(e,f.blocks,cfg)
        results[Path(path).name]={'sha256':sha256(path),**duration_recall(m.labels,pred,m.supervision_mask & m.boundary_mask,m.tracking.blocks)}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'match':m.match_id,'results':results},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(results))


if __name__=='__main__': main()
