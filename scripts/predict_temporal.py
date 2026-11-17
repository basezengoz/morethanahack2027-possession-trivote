"""Generate a private target-match CSV from the fixed research checkpoint, without GT."""
import argparse
import hashlib
import json
from pathlib import Path

import games
import numpy as np
import torch
from possession.decode import decode_labels
from possession.features import build_features
from possession.io import load_tracking
from possession.model import SCHEMA, infer_evidence
from possession.predict import (apply_published_dead_intervals, load_config, sha256,
                                source_provenance, write_submission)
from possession.quality import clean_tracking
from possession.tagging import load_tagging
from possession.teacher import build_evidence
from possession.train import load_checkpoint, zero_tags

CHECKPOINT_SHA256 = 'b6660d662a0c5ad4d11674f7a68ddfe23a1c88b85554966e87c885264f10afed'


def run_temporal(tracking, tags, config, model, normalizer, dead_head_trained, device='cpu'):
    """Apply real tags to rules; provide zero tags to the as-trained TCN."""
    clean = clean_tracking(tracking, config.get('quality', {}), restart=tags.restart)
    features = build_features(clean, config.get('features', {}))
    baseline = build_evidence(features, tags, config)
    temporal = infer_evidence(model, features, zero_tags(len(tracking.frame)), baseline,
                              normalizer, device, dead_head_trained=dead_head_trained)
    if not dead_head_trained and not np.array_equal(temporal.state_scores[:, 2], baseline.state_scores[:, 2]):
        raise ValueError('untrained dead head changed rule evidence')
    for key in ('change', 'stop', 'restart'):
        if not np.array_equal(getattr(temporal, key), getattr(baseline, key)):
            raise ValueError('transition evidence changed')
    labels = decode_labels(temporal, features.blocks, config)
    return apply_published_dead_intervals(labels, features.blocks, config.get('published_dead_intervals', True))


def snapshot(config, checkpoint, inputs):
    source = source_provenance()
    source['source_files_sha256']['scripts/predict_temporal.py'] = sha256(__file__)
    return {'config_sha256': sha256(config), 'checkpoint_sha256': sha256(checkpoint),
            'inputs_sha256': {name: sha256(path) for name, path in inputs.items()},
            'source_files_sha256': source['source_files_sha256']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/rules.yaml')
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--checkpoint', default='data/runs/training/temporal.pt')
    parser.add_argument('--expected-checkpoint-sha256', default=CHECKPOINT_SHA256,
                        help='SHA256 of the trusted local checkpoint; defaults to the fixed first experiment')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--allow-trained-dead-head', action='store_true',
                        help='explicitly use the checkpoint trained D head for this research run')
    parser.add_argument('--output', default='data/runs/public_pipeline_check_tcn/inputs/temporal.csv')
    parser.add_argument('--manifest', default='data/runs/public_pipeline_check_tcn/inputs/temporal.manifest.json')
    args = parser.parse_args(argv)
    allowed = Path('data/runs').resolve()
    for name in ('output', 'manifest'):
        if not Path(getattr(args, name)).resolve().is_relative_to(allowed):
            raise ValueError(name + ' must be under private data/runs')
    if Path(args.output).resolve() == Path(args.manifest).resolve():
        raise ValueError('output and manifest paths must differ')
    paths = games.asset_paths(data_dir=args.data_dir)
    inputs = {name: Path(paths[name]) for name in ('atd_meta', 'atd_raw', 'smart_tagging')}
    before = snapshot(args.config, args.checkpoint, inputs)
    if before['checkpoint_sha256'] != args.expected_checkpoint_sha256:
        raise ValueError('checkpoint SHA256 differs from the declared checkpoint')
    config = load_config(args.config)
    model, norm, checkpoint = load_checkpoint(args.checkpoint, args.device)
    dead_trained = bool(checkpoint['stats']['dead_head_trained'])
    if dead_trained and not args.allow_trained_dead_head:
        raise ValueError('trained D head requires explicit research policy')
    tracking = load_tracking(args.data_dir)
    tags = load_tagging(args.data_dir, len(tracking.frame), tracking.fps, config.get('tagging', {}))
    labels = run_temporal(tracking, tags, config, model, norm, dead_trained, args.device)
    after = snapshot(args.config, args.checkpoint, inputs)
    if before != after:
        raise RuntimeError('source, config, checkpoint or input changed during inference')
    write_submission(args.output, labels)
    schema_hash = hashlib.sha256(json.dumps(checkpoint['schema'], separators=(',', ':')).encode()).hexdigest()
    norm_hash = hashlib.sha256(norm.mean.tobytes() + norm.scale.tobytes()).hexdigest()
    manifest = {'inference_inputs': before, 'output_sha256': sha256(args.output),
                'output': str(Path(args.output).resolve()), 'schema_sha256': schema_hash,
                'normalizer_sha256': norm_hash, 'device': args.device, 'torch_version': torch.__version__,
                'dead_head_trained': dead_trained, 'model_input_tag_policy': 'zero_unseen_training_tags',
                'teacher_tag_policy': 'provided_smart_tagging',
                'dead_evidence_policy': 'trained_D_with_rule_transitions' if dead_trained else 'baseline_rule_qD_and_transitions'}
    path = Path(args.manifest); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    print(f'Temporal submission written: {args.output}; manifest: {path}')


if __name__ == '__main__':
    main()
