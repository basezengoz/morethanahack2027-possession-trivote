"""Run match-disjoint DFL cross-validation with bounded parallel training."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

GAMES=('J03WN1','J03WMX','J03WPY','J03WQQ','J03WOY','J03WOH','J03WR9')


def build_folds(games=GAMES):
    if len(games)<3 or len(set(games))!=len(games): raise ValueError('at least three unique matches required')
    return [{'test':game,'validation':games[(i+1)%len(games)],
             'train':[g for g in games if g not in (game,games[(i+1)%len(games)])]}
            for i,game in enumerate(games)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,help='new run name (letters, numbers, underscore)')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--epochs',type=int,default=6)
    parser.add_argument('--plan-only',action='store_true')
    a=parser.parse_args()
    if not a.run.replace('_','').isalnum() or a.workers<1 or a.epochs<1: raise ValueError('invalid run configuration')
    folds=build_folds()
    plan={'folds':folds,'epochs':a.epochs,'workers':a.workers,
          'methods':{'base':{'feature_version':1,'observation_pattern':False},
                     'context_masked':{'feature_version':2,'observation_pattern':True}},
          'seed':2032,'learning_rate':.0003,'metrica_train':['1','2'],
          'selection':'each fold uses validation match only; clean for base, mean clean and observation masked for context_masked',
          'evaluation':'per held-out match, clean and transferred target-observation mask; no public GT',
          'limitations':['same seven matches repeatedly rotated; aggregate estimates are not an untouched final test',
                         'DFL tracking and label semantics differ from target; target observation mask contains no state labels']}
    if a.plan_only:
        print(json.dumps(plan,indent=2)); return
    root=Path('data/runs')/a.run; checkpoint_root=Path('data/runs/training')/a.run
    if root.exists() or checkpoint_root.exists(): raise ValueError('run name already exists')
    root.mkdir(parents=True); checkpoint_root.mkdir(parents=True)
    for name in ('scripts/train_dfl_multi.py','scripts/evaluate_dfl_masked.py','src/possession/train.py',
                 'src/possession/model.py','src/possession/model_v2.py','src/possession/dfl_data.py',
                 'src/possession/observation_masks.py','configs/rules.yaml'):
        plan.setdefault('source_sha256',{})[name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    lock=threading.Lock(); env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    def train(job):
        fold,method=job; test=fold['test']; checkpoint=checkpoint_root/(test+'_'+method+'.pt')
        cmd=[sys.executable,'-u','scripts/train_dfl_multi.py','--train',*fold['train'],
             '--metrica-train','1','2','--validation',fold['validation'],'--test',test,
             '--output',str(checkpoint),'--epochs',str(a.epochs),'--learning-rate','0.0003',
             '--feature-version','1' if method=='base' else '2','--device','cuda']
        if method=='context_masked': cmd += ['--observation-pattern-data-dir','data']
        with (root/(test+'_'+method+'.log')).open('w',encoding='utf-8') as log:
            completed=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=creationflags)
        result={'test':test,'method':method,'exit_code':completed.returncode,'checkpoint':str(checkpoint)}
        with lock: print(json.dumps({'training_complete':result}),flush=True)
        return result
    jobs=[(f,m) for f in folds for m in plan['methods']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        results=list(pool.map(train,jobs))
    (root/'training_results.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    failed=[r for r in results if r['exit_code']]
    if failed: raise RuntimeError('training failures: '+json.dumps(failed))
    # Evaluate each outer test only after all per-fold checkpoints are fixed.
    for fold in folds:
        game=fold['test']; report=root/(game+'_evaluation.json')
        cmd=[sys.executable,'-u','scripts/evaluate_dfl_masked.py',
             '--positions',f'data/runs/idsse/{game}_positions.xml','--info',f'data/runs/idsse/{game}_info.xml',
             '--checkpoint',str(checkpoint_root/(game+'_base.pt')),
             '--checkpoint',str(checkpoint_root/(game+'_context_masked.pt')),
             '--observation-pattern-data-dir','data','--output',str(report)]
        with (root/(game+'_evaluation.log')).open('w',encoding='utf-8') as log:
            completed=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=creationflags)
        if completed.returncode: raise RuntimeError('evaluation failed for '+game)
        print(json.dumps({'evaluation_complete':game}),flush=True)
    summary={}
    for method in plan['methods']:
        rows=[]
        for fold in folds:
            game=fold['test']; report=json.loads((root/(game+'_evaluation.json')).read_text(encoding='utf-8'))
            rows.append({'match':game,**report['results'][game+'_'+method+'.pt']})
        summary[method]={'folds':rows,
            'mean_clean_spell_f1':sum(x['clean']['spell_f1']['mean_f1'] for x in rows)/len(rows),
            'mean_masked_spell_f1':sum(x['masked']['spell_f1']['mean_f1'] for x in rows)/len(rows)}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'summary':str(root/'summary.json')}),flush=True)


if __name__=='__main__': main()
