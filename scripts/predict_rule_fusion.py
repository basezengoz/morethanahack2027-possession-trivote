"""Generate a frozen experimental pass-fusion target CSV without reading ground truth."""
import argparse
import json
from pathlib import Path
import numpy as np
import games

from possession.decode import decode_labels
from possession.features import build_features
from possession.io import load_tracking
from possession.predict import (apply_published_dead_intervals, load_config, make_manifest,
                                sha256, write_submission)
from possession.quality import clean_tracking
from possession.rule_based import dead_ball
from possession.rule_based.loader import load_match, load_tracking as load_rule_tracking
from possession.rule_fusion import combine_evidence, cues_from_rule_features
from possession.tagging import load_tagging
from possession.teacher import build_evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--selection', required=True)
    p.add_argument('--data-dir', default=None)
    args = p.parse_args()
    out = Path(args.output).resolve()
    if not out.is_relative_to(Path('data/runs').resolve()):
        raise ValueError('experimental output must be under data/runs')
    selection_path = Path(args.selection)
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if selection['selected'] != 'pass':
        raise ValueError('this entry point requires the frozen pass candidate')
    for path, expected in selection['source_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('source/config differs from frozen selection: ' + path)
    cfg = load_config('configs/rules.yaml')
    t = load_tracking(args.data_dir)
    tags = load_tagging(args.data_dir, len(t.frame), t.fps, cfg.get('tagging', {}))
    feature = build_features(clean_tracking(t, cfg.get('quality', {}), tags.restart), cfg.get('features', {}))
    base = build_evidence(feature, tags, cfg)
    _, _, ids, _ = load_rule_tracking(data_dir=args.data_dir)
    if not np.array_equal(ids, np.arange(1, len(t.frame) + 1)):
        raise ValueError('kural tabanlı yöntem tracking frame IDs must be exactly 1..N in order')
    m = load_match(data_dir=args.data_dir, with_homography=False)
    pf = dead_ball.features(m)
    passes = np.zeros((len(t.frame), 2), np.float32)
    restart = np.zeros(len(t.frame), np.float32)
    for start, end in t.blocks:
        local = {'passes': [(a-start, b-start, team) for a, b, team in pf['passes']
                            if start <= a <= b < end],
                 'out': pf['out'][start:end], 'setpiece': pf['setpiece'][start:end]}
        passes[start:end], restart[start:end] = cues_from_rule_features(local, end-start)
    evidence = combine_evidence(base, passes, restart, use_pass=True, use_restart=False)
    labels = decode_labels(evidence, feature.blocks, cfg)
    apply_published_dead_intervals(labels, feature.blocks, cfg.get('published_dead_intervals', True))
    write_submission(out, labels)
    paths = games.asset_paths(data_dir=args.data_dir)
    manifest = make_manifest('configs/rules.yaml', {k: Path(paths[k]) for k in ('atd_meta', 'atd_raw', 'smart_tagging')}, out)
    manifest['experiment'] = {'method': 'pass', 'selection_sha256': sha256(selection_path),
                              'entrypoint_sha256': sha256(Path(__file__).resolve()),
                              'pass_support_frames': int(np.any(passes > 0, axis=1).sum()),
                              'note': 'experimental candidate; no public ground truth used'}
    path = out.with_suffix('.manifest.json')
    path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Experimental CSV:', out, 'manifest:', path)


if __name__ == '__main__':
    main()
