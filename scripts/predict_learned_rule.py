"""Generate a private experimental kural tabanlı yöntem+learned-TCN CSV without target labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import games
from possession.features import build_features
from possession.io import load_tracking
from possession.learned_rule_postprocess import cues, tag_mask
from possession.model import infer_evidence
from possession.predict import apply_published_dead_intervals, load_config, make_manifest, sha256, write_submission
from possession.quality import clean_tracking
from possession.rule_based import dead_ball, model as rule_model
from possession.rule_based.loader import load_match, load_tracking as load_rule_tracking
from possession.tagging import load_tagging
from possession.teacher import build_evidence
from possession.train import load_checkpoint, zero_tags


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--weights', required=True)
    p.add_argument('--selection', required=True); p.add_argument('--output', required=True)
    p.add_argument('--data-dir', default=None); p.add_argument('--config', default='configs/rules.yaml')
    a = p.parse_args(); output = Path(a.output).resolve(); private = Path('data/runs').resolve()
    if not output.is_relative_to(private): raise ValueError('output must be under data/runs')
    weights_path = Path(a.weights).resolve(); selection_path = Path(a.selection).resolve()
    if not weights_path.is_relative_to(private) or not selection_path.is_relative_to(private): raise ValueError('private experiment files required')
    weights = json.loads(weights_path.read_text(encoding='utf-8'))
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if selection.get('selected') != 'learned' or selection.get('weights_sha256') != sha256(weights_path):
        raise ValueError('frozen selection must approve these learned weights')
    if sha256(a.checkpoint) != weights['training'][0]['checkpoint']: raise ValueError('checkpoint differs from training')
    for path, expected in weights['source_sha256'].items():
        if sha256(path) != expected: raise ValueError('fitted source changed: '+path)
    cfg = load_config(a.config); tracking = load_tracking(a.data_dir)
    model, norm, ck = load_checkpoint(a.checkpoint, 'cpu')
    if not ck['stats']['dead_head_trained']: raise ValueError('trained D head required')
    tags = load_tagging(a.data_dir, len(tracking.frame), tracking.fps, cfg.get('tagging', {}))
    clean = clean_tracking(tracking, cfg.get('quality', {}), tags.restart)
    features = build_features(clean, cfg.get('features', {}))
    base = build_evidence(features, tags, cfg)
    temporal = infer_evidence(model, features, zero_tags(len(tracking.frame)), base, norm, dead_head_trained=True)
    _, _, ids, _ = load_rule_tracking(data_dir=a.data_dir)
    if not np.array_equal(ids, np.arange(1, len(tracking.frame)+1)): raise ValueError('invalid ATD frame IDs')
    m = load_match(data_dir=a.data_dir, with_homography=False)
    if m.n_frames != len(tracking.frame) or not np.array_equal(m.team, tracking.slot_team):
        raise ValueError('kural tabanlı yöntem and TCN tracking do not align')
    f = dead_ball.features(m); llr = dead_ball.dead_evidence(m, f)
    dead = dead_ball.detect_dead(m, llr.copy(), f)
    rule_labels = rule_model.build_labels(m, f, dead)[0]
    observed = np.isfinite(tracking.ball_xy_raw).all(axis=1)
    team, dead_cue = cues(temporal.state_scores, rule_labels, observed, weights['weights'], tag_mask(m))
    team[m.half == 0] = 0; dead_cue[m.half == 0] = 0
    dead = dead_ball.detect_dead(m, llr + dead_cue, f)
    labels = rule_model.build_labels(m, f, dead, team)[0]
    apply_published_dead_intervals(labels, tracking.blocks, cfg.get('published_dead_intervals', True))
    write_submission(output, labels)
    paths = games.asset_paths(data_dir=a.data_dir)
    manifest = make_manifest(a.config, {k: Path(paths[k]) for k in ('atd_meta', 'atd_raw', 'smart_tagging')}, output)
    manifest['experiment'] = {'checkpoint_sha256': sha256(a.checkpoint), 'weights_sha256': sha256(weights_path),
        'selection_sha256': sha256(selection_path), 'entrypoint_sha256': sha256(__file__),
        'rule_sources_sha256': {str(p): sha256(p) for p in (Path('src/possession/rule_based/model.py'),
            Path('src/possession/rule_based/dead_ball.py'), Path('src/possession/rule_based/loader.py'))},
        'tag_policy': 'kural tabanlı yöntem uses provided Smart Tagging; model input tags zero; learned cue suppressed on tag intervals',
        'note': 'experimental; target GT not read'}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print('Experimental CSV:', output)


if __name__ == '__main__': main()
