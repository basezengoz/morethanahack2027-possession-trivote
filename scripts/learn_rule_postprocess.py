"""Fit and evaluate a learned TCN correction within the kural tabanlı yöntem method on DFL data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from possession.decode import decode_labels
from possession.dfl_data import load_dfl
from possession.external_evaluation import evaluate_external
from possession.features import build_features
from possession.io import load_tracking
from possession.learned_rule_postprocess import inputs, cues, tag_mask
from possession.model import infer_evidence
from possession.observation_masks import transfer_ball_mask
from possession.predict import load_config, sha256
from possession.rule_based_tcn_evidence import tcn_cues
from possession.quality import clean_tracking
from possession.rule_based import dead_ball, model as rule_model
from possession.teacher import build_evidence
from possession.train import load_checkpoint, zero_tags
from scripts.compare_rule_methods import dfl_as_rule_based_match, dfl_goalkeepers
from scripts.analyze_short_possessions import duration_recall


def private(path):
    path = Path(path).resolve()
    if not path.is_relative_to(Path('data/runs').resolve()): raise ValueError('private data/runs path required')
    return path


def calculate(tracking, goalies, cfg, model, norm, trained_dead, weights=None):
    n = len(tracking.frame)
    tags = zero_tags(n)
    features = build_features(clean_tracking(tracking, cfg.get('quality', {}), tags.restart), cfg.get('features', {}))
    base = build_evidence(features, tags, cfg)
    temporal = infer_evidence(model, features, tags, base, norm, dead_head_trained=trained_dead)
    observed = np.isfinite(tracking.ball_xy_raw).all(axis=1)
    predictions = {k: np.full(n, 'X', dtype='<U1') for k in ('rule_based', 'fixed', 'learned')}
    predictions['tcn'] = decode_labels(temporal, tracking.blocks, cfg)
    for start, end in tracking.blocks:
        m = dfl_as_rule_based_match(tracking, start, end, goalies)
        f = dead_ball.features(m)
        llr = dead_ball.dead_evidence(m, f)
        plain_dead = dead_ball.detect_dead(m, llr.copy(), f)
        plain = rule_model.build_labels(m, f, plain_dead)[0]
        predictions['rule_based'][start:end] = plain
        if weights is not None:
            fixed_team, fixed_dead = tcn_cues(temporal.state_scores[start:end], observed[start:end])
            learned_team, learned_dead = cues(temporal.state_scores[start:end], plain, observed[start:end], weights, tag_mask(m))
            for name, team, d in [('fixed', fixed_team, fixed_dead), ('learned', learned_team, learned_dead)]:
                dead = dead_ball.detect_dead(m, llr + d, f)
                predictions[name][start:end] = rule_model.build_labels(m, f, dead, team)[0]
    return temporal.state_scores, observed, predictions


def load_views(args):
    m = load_dfl(args.positions, args.info)
    model, norm, checkpoint = load_checkpoint(args.checkpoint, 'cpu')
    if not checkpoint['stats']['dead_head_trained']: raise ValueError('trained D head required')
    goalies = dfl_goalkeepers(args.positions, args.info, m.tracking)
    pattern = load_tracking(args.pattern_data_dir)
    masked, dropped = transfer_ball_mask(m.tracking, pattern, offset=7919)
    del pattern
    return m, model, norm, checkpoint, goalies, masked, dropped


def cache(args):
    output = private(args.output)
    m, model, norm, ck, goalies, masked, dropped = load_views(args)
    train = {str(x).split(':')[0] for x in ck.get('train_matches', [])}
    if m.match_id not in train: raise ValueError('cache mode expects a declared TCN training match')
    cfg = load_config('configs/rules.yaml'); arrays = {}
    for view, t in [('clean', m.tracking), ('masked', masked)]:
        scores, observed, predictions = calculate(t, goalies, cfg, model, norm, True)
        arrays[view + '_scores'] = scores; arrays[view + '_observed'] = observed
        arrays[view + '_rule_based'] = predictions['rule_based']
    arrays.update(labels=m.labels, known=m.supervision_mask, match=np.array(m.match_id),
                  checkpoint=np.array(sha256(args.checkpoint)))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(json.dumps({'match': m.match_id, 'checkpoint': sha256(args.checkpoint), 'masked_frames': int(dropped.sum()), 'output': str(output)}))


def fit(args):
    output = private(args.output); rng = np.random.default_rng(2027); txs=[]; dxs=[]; ys=[]; manifests=[]
    for path in args.cache:
        path = private(path)
        with np.load(path, allow_pickle=False) as data:
            labels = data['labels']; known = data['known'].astype(bool)
            for view in ('clean', 'masked'):
                tx, dx = inputs(data[view+'_scores'], data[view+'_rule_based'], data[view+'_observed'])
                indices = np.flatnonzero(known & np.isin(labels, [0, 1, 2]) & (data[view+'_rule_based'] != 'X'))
                if len(indices) > 50000: indices = rng.choice(indices, 50000, replace=False)
                txs.append(tx[indices]); dxs.append(dx[indices]); ys.append(labels[indices])
            manifests.append({'match': str(data['match']), 'checkpoint': str(data['checkpoint']), 'sha256': sha256(path)})
    if len({m['match'] for m in manifests}) != len(manifests) or len({m['checkpoint'] for m in manifests}) != 1:
        raise ValueError('unique matches and one checkpoint required')
    y = np.concatenate(ys); team_x = np.concatenate(txs); dead_x = np.concatenate(dxs)
    if set(np.unique(y)) != {0, 1, 2}: raise ValueError('all three supervised states required')
    def optimize(x, target):
        x = torch.tensor(x, dtype=torch.float32); target = torch.tensor(target, dtype=torch.float32)
        w = torch.nn.Parameter(torch.zeros(x.shape[1])); opt = torch.optim.LBFGS([w], max_iter=60, line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w, target) + .01 * (w*w).sum()
            loss.backward(); return loss
        opt.step(closure); return w.detach().numpy().tolist()
    live = y != 2
    weights = {'team': optimize(team_x[live], y[live] == 0), 'dead': optimize(dead_x, y == 2)}
    report = {'weights': weights, 'training': manifests, 'n_samples': int(len(y)),
              'class_counts': {str(c): int((y == c).sum()) for c in (0, 1, 2)}, 'seed': 2027,
              'protocol_sha256': sha256('docs/learned_rule_based_tcn_postprocess.md'),
              'source_sha256': {p: sha256(p) for p in ('scripts/learn_rule_postprocess.py', 'src/possession/learned_rule_postprocess.py', 'src/possession/rule_based/model.py', 'src/possession/rule_based/dead_ball.py')},
              'limitation': 'TCN training-match predictions, not out-of-fold; DFL has no Smart Tagging'}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'weights': weights, 'counts': report['class_counts']}))


def evaluate(args):
    output = private(args.output); fit_path = private(args.weights)
    fitted = json.loads(fit_path.read_text(encoding='utf-8'))
    if sha256(args.checkpoint) != fitted['training'][0]['checkpoint']: raise ValueError('TCN checkpoint differs from fitting')
    m, model, norm, ck, goalies, masked, dropped = load_views(args)
    if m.match_id in {str(x).split(':')[0] for x in ck.get('train_matches', [])}:
        raise ValueError('evaluation match used in TCN training')
    if m.match_id in {x['match'] for x in fitted['training']}: raise ValueError('evaluation match used in postprocessor training')
    cfg = load_config('configs/rules.yaml'); all_predictions = {}
    for view, t in [('clean', m.tracking), ('masked', masked)]:
        _, _, all_predictions[view] = calculate(t, goalies, cfg, model, norm, True, fitted['weights'])
    results = {}
    for view, predictions in all_predictions.items():
        results[view] = {}
        for name, labels in predictions.items():
            metric = evaluate_external(m, labels)
            metric['duration_recall'] = duration_recall(m.labels, labels, m.supervision_mask & m.boundary_mask, m.tracking.blocks)
            results[view][name] = metric
    report = {'match': m.match_id, 'reference': m.provenance, 'postprocessor_sha256': sha256(fit_path),
              'checkpoint_sha256': sha256(args.checkpoint), 'mask_sha256': hashlib.sha256(dropped.tobytes()).hexdigest(),
              'masked_frames': int(dropped.sum()), 'results': results,
              'limitations': ['DFL has no Smart Tagging; tag gate unmeasured on real tags', 'DFL tracking and labels differ from target',
                              'TCN training predictions used to fit postprocessor', 'previously used development/control match']}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({view: {name: round(m['spell_f1']['mean_f1'], 6) for name, m in models.items()} for view, models in results.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='mode', required=True)
    for name in ('cache', 'evaluate'):
        p = sub.add_parser(name)
        for key in ('positions', 'info', 'checkpoint', 'output'): p.add_argument('--'+key, required=True)
        p.add_argument('--pattern-data-dir', default='data')
        if name == 'evaluate': p.add_argument('--weights', required=True)
    p = sub.add_parser('fit'); p.add_argument('--cache', nargs='+', required=True); p.add_argument('--output', required=True)
    args = parser.parse_args(); {'cache': cache, 'fit': fit, 'evaluate': evaluate}[args.mode](args)


if __name__ == '__main__': main()
