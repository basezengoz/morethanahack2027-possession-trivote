"""Compare fixed rule-based geometry and a trusted temporal checkpoint on the same proxy."""
import argparse
import json
from pathlib import Path
import numpy as np
from possession.contracts import TaggingEvidence
from possession.external_data import load_metrica_csv
from possession.external_evaluation import evaluate_external
from possession.quality import clean_tracking
from possession.features import build_features
from possession.teacher import build_evidence
from possession.decode import decode_labels
from possession.model import infer_evidence
from possession.train import load_checkpoint
from possession.predict import load_config, sha256
if __package__:
    from scripts.evaluate_external import source_provenance
else:
    from evaluate_external import source_provenance


def validate_split(match_id, split, checkpoint):
    train = {str(x) for x in checkpoint['train_matches']}
    val = {str(x) for x in checkpoint['validation_matches']}
    if not train or not val or train & val:
        raise ValueError('checkpoint must contain disjoint train and validation matches')
    match_id = str(match_id)
    if (split == 'train' and match_id not in train or
        split == 'validation' and match_id not in val or
        split == 'test' and match_id in train | val):
        raise ValueError('requested match does not match checkpoint split')


def compare(match, config, model, norm, checkpoint, device='cpu', tolerance=25):
    tracking = match.tracking
    n = len(tracking.frame)
    tags = TaggingEvidence(np.zeros((n, 2)), np.zeros(n), np.zeros(n))
    clean = clean_tracking(tracking, config.get('quality', {}), tags.restart)
    features = build_features(clean, config.get('features', {}))
    baseline = build_evidence(features, tags, config)
    temporal = infer_evidence(model, features, tags, baseline, norm, device,
                              dead_head_trained=checkpoint['stats']['dead_head_trained'])
    if not checkpoint['stats']['dead_head_trained'] and not np.array_equal(temporal.state_scores[:, 2], baseline.state_scores[:, 2]):
        raise ValueError('untrained D head changed baseline D evidence')
    for key in ('change', 'stop', 'restart'):
        if not np.array_equal(getattr(temporal, key), getattr(baseline, key)):
            raise ValueError('temporal inference changed transition evidence')
    a = decode_labels(baseline, features.blocks, config)
    b = decode_labels(temporal, features.blocks, config)
    rule_based_proxy = evaluate_external(match, a, tolerance)
    return {'rule_based_geometry_proxy': rule_based_proxy,
            'A_geometry_proxy': rule_based_proxy,
            'temporal_proxy': evaluate_external(match, b, tolerance),
            'labels_changed_frames': int(np.count_nonzero(a != b)),
            'D_membership_changed_frames': int(np.count_nonzero((a == 'D') != (b == 'D'))),
            'dead_evidence_note': 'when untrained, qD and transition cues are preserved; decoded D labels can still change',
            'dead_head_trained': bool(checkpoint['stats']['dead_head_trained'])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('match-id', 'home', 'away', 'events', 'checkpoint'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--split', required=True, choices=['train', 'validation', 'test'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--config', default='configs/rules.yaml')
    parser.add_argument('--boundary-tolerance', type=int, default=25)
    parser.add_argument('--output', default='data/runs/external/temporal_comparison.json')
    args = parser.parse_args(argv)
    source = source_provenance()
    root = Path(__file__).resolve().parents[1]
    additional = ('scripts/evaluate_temporal.py', 'src/possession/model.py', 'src/possession/train.py')
    source['files_sha256'].update({name: sha256(root / name) for name in additional})
    cfg = load_config(args.config)
    model, norm, checkpoint = load_checkpoint(args.checkpoint, args.device)
    validate_split(args.match_id, args.split, checkpoint)
    paths = [Path(args.home), Path(args.away), Path(args.events)]
    input_hashes = {p.name: sha256(p) for p in paths}
    recorded = checkpoint['input_sha256']
    if args.split in ('train', 'validation'):
        if recorded.get(str(args.match_id)) != input_hashes:
            raise ValueError('match input hashes differ from checkpoint provenance')
    elif any(set(input_hashes.values()) & set(game.values()) for game in recorded.values()):
        raise ValueError('test inputs overlap checkpoint training or validation inputs')
    match = load_metrica_csv(args.match_id, args.home, args.away, args.events)
    result = compare(match, cfg, model, norm, checkpoint, args.device, args.boundary_tolerance)
    report = {'match_id': match.match_id, 'split': args.split, 'arguments': vars(args),
              'data_provenance': match.provenance, 'source': source,
              'baseline_config_sha256': sha256(args.config),
              'checkpoint': {'sha256': sha256(args.checkpoint), 'schema': checkpoint['schema'],
                             'config': checkpoint['config'], 'config_sha256': checkpoint['config_sha256'],
                             'training_source_sha256': checkpoint['source_sha256'],
                             'training_input_sha256': checkpoint['input_sha256'],
                             'train_matches': checkpoint['train_matches'],
                             'validation_matches': checkpoint['validation_matches'],
                             'seed': checkpoint['config']['seed'], 'stats': checkpoint['stats']},
              'comparison': result, 'rule_based_plus_tags': 'unavailable',
              'A_plus_tags': 'unavailable',
              'limitations': ['event-derived proxy is not possession ground truth',
                              'no independent outer test unless separately supplied',
                              'do not tune or select models on a test split']}
    current = source_provenance()['files_sha256']
    current.update({name: sha256(root / name) for name in additional})
    if source['files_sha256'] != current:
        raise RuntimeError('source files changed during evaluation')
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'Aggregate temporal proxy comparison written: {out}')


if __name__ == '__main__':
    main()
