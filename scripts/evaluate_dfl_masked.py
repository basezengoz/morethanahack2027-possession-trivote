"""Compare fixed checkpoints on clean and deterministically masked DFL tracking."""
import argparse
import json
from pathlib import Path
from possession.dfl_data import load_dfl
from possession.external_evaluation import evaluate_external, mask_ball
from possession.train import load_checkpoint, zero_tags
from possession.predict import load_config, sha256
from possession.quality import clean_tracking
from possession.features import build_features
from possession.teacher import build_evidence
from possession.model import infer_evidence
from possession.decode import decode_labels
from scripts.analyze_short_possessions import duration_recall


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('positions','info','output'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--checkpoint',action='append',required=True)
    parser.add_argument('--fraction',type=float,default=.4)
    parser.add_argument('--block-frames',type=int,default=100)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--observation-pattern-data-dir',default=None)
    args=parser.parse_args()
    out=Path(args.output).resolve()
    if not out.is_relative_to(Path('data/runs').resolve()): raise ValueError('private output required')
    match=load_dfl(args.positions,args.info); cfg=load_config('configs/rules.yaml')
    if args.observation_pattern_data_dir:
        from possession.io import load_tracking
        from possession.observation_masks import transfer_ball_mask
        pattern=load_tracking(args.observation_pattern_data_dir)
        masked,dropped=transfer_ball_mask(match.tracking,pattern,offset=7919)
    else:
        masked,dropped=mask_ball(match.tracking,args.seed,args.fraction,args.block_frames)
    tags=zero_tags(len(match.labels)); views={}
    for name,tracking in (('clean',match.tracking),('masked',masked)):
        features=build_features(clean_tracking(tracking,cfg.get('quality',{}),tags.restart),cfg.get('features',{}))
        views[name]=(features,build_evidence(features,tags,cfg))
    results={}
    for path in args.checkpoint:
        model,norm,checkpoint=load_checkpoint(path,'cpu'); metrics={}
        if match.match_id in {str(x) for x in checkpoint.get('train_matches', [])}:
            raise ValueError('reference match appears in checkpoint training matches')
        if Path(path).name in results: raise ValueError('checkpoint names must be unique')
        for name,(features,base) in views.items():
            evidence=infer_evidence(model,features,tags,base,norm,dead_head_trained=checkpoint['stats']['dead_head_trained'])
            prediction=decode_labels(evidence,features.blocks,cfg)
            result=evaluate_external(match,prediction)
            if result['spell_f1'] is None: raise ValueError('spell coverage unavailable')
            metrics[name]={'spell_f1':result['spell_f1'], 'D':result['classes']['D'],
                           'boundary':result['boundary'], 'macro_f1':result['macro_f1'],
                           'duration_recall':duration_recall(match.labels,prediction,
                               match.supervision_mask & match.boundary_mask,match.tracking.blocks)}
        metrics['mean_clean_masked_spell_f1']=(metrics['clean']['spell_f1']['mean_f1']+metrics['masked']['spell_f1']['mean_f1'])/2
        results[Path(path).name]={'sha256':sha256(path),**metrics}
    report={'match':match.match_id,'reference':match.provenance,'seed':args.seed,
            'requested_fraction':args.fraction,'block_frames':args.block_frames,
            'masked_observed_frames':int(dropped.sum()),'results':results,
            'scope':'development match; synthetic missing data, not a target-match accuracy measurement'}
    if args.observation_pattern_data_dir:
        import games
        paths=games.asset_paths(data_dir=args.observation_pattern_data_dir)
        report['observation_pattern_sha256']={k:sha256(paths[k]) for k in ('atd_meta','atd_raw')}
        report['pattern_offset']=7919
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(results))


if __name__=='__main__': main()
