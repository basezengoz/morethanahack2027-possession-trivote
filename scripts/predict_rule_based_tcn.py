"""Generate an experimental kural tabanlı yöntem + TCN submission using tactical tracking and tags."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import games
from possession.features import build_features
from possession.io import load_tracking
from possession.model import infer_evidence
from possession.predict import apply_published_dead_intervals, load_config, make_manifest, sha256, text_sha256, source_provenance, write_submission
from possession.rule_based_tcn_evidence import tcn_cues
from possession.quality import clean_tracking
from possession.rule_based import dead_ball, model as rule_model
from possession.rule_based.loader import load_match, load_tracking as load_rule_tracking
from possession.tagging import load_tagging
from possession.teacher import build_evidence
from possession.train import load_checkpoint, zero_tags


def checkpoint_policy(actual, tested, explicit):
    """Allow an explicitly identified user checkpoint without calling it validated."""
    if explicit is not None and explicit.lower() != actual.lower():
        raise ValueError('checkpoint differs from explicitly expected SHA256')
    if actual.lower() != tested.lower() and explicit is None:
        raise ValueError('checkpoint differs from experiment; provide its explicit SHA256')
    return 'experiment_checkpoint' if actual.lower() == tested.lower() else 'user_selected_untested_combination'


def snapshot(args, inputs):
    src = source_provenance()['source_files_sha256']
    for p in ('scripts/predict_rule_based_tcn.py', 'src/possession/rule_based/model.py',
              'src/possession/rule_based/dead_ball.py', 'src/possession/rule_based/loader.py'):
        src[p] = sha256(p)
    return {'source': src, 'config': sha256(args.config), 'checkpoint': sha256(args.checkpoint),
            'frozen_config': sha256(args.frozen_config) if args.frozen_config else None,
            'selection': sha256(args.selection) if args.selection else None,
            'report': sha256(args.development_report) if args.development_report else None,
            'inputs': {k: sha256(v) for k, v in inputs.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--selection')
    p.add_argument('--development-report'); p.add_argument('--frozen-config'); p.add_argument('--output', required=True)
    p.add_argument('--expected-checkpoint-sha256', default=None,
                   help='explicitly identify an alternate checkpoint; fusion with it is not validated by the old report')
    p.add_argument('--data-dir', default=None); p.add_argument('--config', default='configs/rules.yaml')
    a = p.parse_args(); private = Path('data/runs').resolve(); output = Path(a.output).resolve()
    if not output.is_relative_to(private) or output.suffix.lower() != '.csv':
        raise ValueError('CSV output must be under data/runs')
    if a.frozen_config:
        if a.selection or a.development_report: p.error('--frozen-config cannot be combined with private report options')
        frozen = json.loads(Path(a.frozen_config).read_text(encoding='utf-8'))
        if frozen.get('selected') != 'combined': raise ValueError('frozen selection must be combined')
        report = {'checkpoint_sha256': frozen['development_checkpoint_sha256'], 'source_sha256': frozen['source_sha256']}
        report_hash = frozen['development_report_sha256']
    else:
        if not a.selection or not a.development_report: p.error('provide --frozen-config or both private report options')
        for path in (a.selection, a.development_report):
            if not Path(path).resolve().is_relative_to(private): raise ValueError('selection and report must be private')
        selection = json.loads(Path(a.selection).read_text(encoding='utf-8'))
        report = json.loads(Path(a.development_report).read_text(encoding='utf-8'))
        if selection.get('selected') != 'combined' or selection.get('development_report_sha256') != sha256(a.development_report):
            raise ValueError('selection does not match combined development report')
        report_hash = sha256(a.development_report)
    policy = checkpoint_policy(sha256(a.checkpoint), report['checkpoint_sha256'], a.expected_checkpoint_sha256)
    for path, expected in report['source_sha256'].items():
        actual = text_sha256(path) if a.frozen_config else sha256(path)
        if actual != expected: raise ValueError('experiment source changed: '+path)
    paths = games.asset_paths(data_dir=a.data_dir)
    inputs = {k: Path(paths[k]) for k in ('atd_meta', 'atd_raw', 'smart_tagging')}
    before = snapshot(a, inputs)
    cfg = load_config(a.config); tracking = load_tracking(a.data_dir)
    model, norm, ck = load_checkpoint(a.checkpoint, 'cpu')
    if not ck['stats']['dead_head_trained']: raise ValueError('trained D head required')
    tags = load_tagging(a.data_dir, len(tracking.frame), tracking.fps, cfg.get('tagging', {}))
    clean = clean_tracking(tracking, cfg.get('quality', {}), tags.restart)
    features = build_features(clean, cfg.get('features', {}))
    baseline = build_evidence(features, tags, cfg)
    tcn = infer_evidence(model, features, zero_tags(len(tracking.frame)), baseline, norm, dead_head_trained=True)
    _, _, ids, _ = load_rule_tracking(data_dir=a.data_dir)
    if not np.array_equal(ids, np.arange(1, len(tracking.frame) + 1)): raise ValueError('invalid ATD frame IDs')
    m = load_match(data_dir=a.data_dir, with_homography=False)
    if m.n_frames != len(tracking.frame) or not np.array_equal(m.team, tracking.slot_team):
        raise ValueError('kural tabanlı yöntem and TCN tracking do not align')
    observed = np.isfinite(tracking.ball_xy_raw).all(axis=1)
    team, dead_cue = tcn_cues(tcn.state_scores, observed)
    team[m.half == 0] = 0; dead_cue[m.half == 0] = 0
    f = dead_ball.features(m)
    dead = dead_ball.detect_dead(m, dead_ball.dead_evidence(m, f) + dead_cue, f)
    labels = rule_model.build_labels(m, f, dead, team)[0]
    apply_published_dead_intervals(labels, tracking.blocks, cfg.get('published_dead_intervals', True))
    if snapshot(a, inputs) != before: raise RuntimeError('source or inputs changed during inference')
    write_submission(output, labels)
    manifest = make_manifest(a.config, inputs, output)
    manifest['experiment'] = {'frozen_inputs': before, 'checkpoint_sha256': before['checkpoint'],
        'selection_sha256': before['selection'], 'development_report_sha256': report_hash,
        'frozen_config_sha256': before['frozen_config'],
        'checkpoint_policy': policy, 'development_checkpoint_sha256': report['checkpoint_sha256'],
        'tag_policy': 'kural tabanlı yöntem uses provided Smart Tagging; TCN inputs use zero tags as during training',
        'tcn_cues': 'bounded H/A and D support on raw observed-ball frames',
        'limitations': 'experimental; DFL development lacks Smart Tagging; target accuracy unknown'}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print('Experimental kural tabanlı yöntem + TCN CSV:', output)


if __name__ == '__main__':
    # Windows consoles/pipes default to cp1252, which cannot print the Turkish status line.
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    main()
