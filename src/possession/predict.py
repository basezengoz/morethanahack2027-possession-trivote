"""Generate a deterministic rule-based submission and private reproducibility manifest."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
import games
import starter.load_smart_tagging
from .io import load_tracking
from .tagging import load_tagging
from .quality import clean_tracking
from .features import build_features
from .teacher import build_evidence
from .decode import decode_labels


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    permitted = {"quality", "features", "tagging", "teacher", "decode", "published_dead_intervals"}
    if set(config) - permitted:
        raise ValueError("unknown config sections: " + ", ".join(sorted(set(config) - permitted)))
    for key in ("quality", "features", "tagging", "teacher", "decode"):
        if not isinstance(config.get(key, {}), dict):
            raise ValueError(f"{key} must be a mapping")
    if not isinstance(config.get("published_dead_intervals", True), bool):
        raise ValueError("published_dead_intervals must be boolean")
    return config


def apply_published_dead_intervals(labels, blocks, enabled=True):
    """Apply the published second-half pre-kickoff dead period, clipped to block."""
    if enabled:
        official = games.half_video_blocks(games.GAME)
        second_start = next(start for half, start, _ in official if half == 2)
        kickoff = games.anchor_video_frame(games.GAMES[games.GAME]["kickoff_2nd"])
        for start, end in blocks:
            left, right = max(start, second_start), min(end, kickoff)
            if left < right:
                labels[left:right] = "D"
    return labels


def run_pipeline(config, data_dir):
    """Run inference; return labels and actual input paths for provenance."""
    tracking = load_tracking(data_dir)
    tags = load_tagging(data_dir, len(tracking.frame), tracking.fps, config.get("tagging", {}))
    clean = clean_tracking(tracking, config.get("quality", {}), restart=tags.restart)
    features = build_features(clean, config.get("features", {}))
    evidence = build_evidence(features, tags, config)
    labels = decode_labels(evidence, features.blocks, config)
    apply_published_dead_intervals(labels, features.blocks, config.get("published_dead_intervals", True))
    paths = games.asset_paths(data_dir=data_dir)
    return labels, {name: Path(paths[name]) for name in ("atd_meta", "atd_raw", "smart_tagging")}


def write_submission(path, labels):
    path = Path(path)
    if np.asarray(labels).ndim != 1 or not np.isin(labels, ["H", "A", "D", "X"]).all():
        raise ValueError("labels must be a one-dimensional H/A/D/X array")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("frame,label\n")
        stream.writelines(f"{i},{label}\n" for i, label in enumerate(labels))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(path):
    """Hash text independent of the checkout's CRLF/LF convention."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def source_provenance():
    package = Path(__file__).resolve().parent
    source_paths = list(package.glob("*.py")) + [Path(games.__file__).resolve(), Path(starter.load_smart_tagging.__file__).resolve()]
    root = package.parent.parent
    for name in ("pyproject.toml", "uv.lock"):
        p = root / name
        if p.is_file():
            source_paths.append(p)
    hashes = {("possession/" + p.name if p.parent == package else
               "starter/" + p.name if p.parent.name == "starter" else p.name): sha256(p)
              for p in sorted(set(source_paths))}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    def git(*args):
        try:
            return subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=all")
    return {"git_revision": revision, "git_dirty": None if status is None else bool(status),
            "source_sha256": fingerprint, "source_files_sha256": hashes}


def make_manifest(config_path, input_paths, output):
    versions = {}
    for name in ("more-than-a-hack-2027", "numpy", "pandas", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"config": {"path": str(Path(config_path).resolve()), "sha256": sha256(config_path)},
            "inputs": {name: {"path": str(p.resolve()), "sha256": sha256(p)} for name, p in input_paths.items()},
            "output": {"path": str(Path(output).resolve()), "sha256": sha256(output)},
            "source": source_provenance(), "python": platform.python_version(), "packages": versions,
            "seed": None, "device": "cpu", "checkpoint": None}


ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    'v6': ('data/runs/training/dfl_v6.pt', '3afb66101c2cac6ab05305eb3a1f197882faf5fbeead56decfe498705807ddaf'),
    'v5': ('data/runs/training/dfl_v5_observation_pattern.pt', '9bb38a2ecd116265f985c3777cd0c15834f4a4ba43233ce2099aa7e8a5b55c29'),
    'v4': ('data/runs/training/dfl_v4_missing100.pt', '0750944c767aadbd8033660de1438fe8de54d5d1fec7df2c48b18f622d4919ed'),
}


def run_rule_based(config, data_dir):
    """Run the new rule-based method with Smart Tagging, without TCN."""
    from .rule_based import dead_ball, model
    from .rule_based.loader import load_match, load_tracking as load_rule_tracking
    tracking = load_tracking(data_dir)
    _, _, ids, _ = load_rule_tracking(data_dir=data_dir)
    if not np.array_equal(ids, np.arange(1, len(tracking.frame) + 1)):
        raise ValueError('invalid ATD frame IDs')
    match = load_match(data_dir=data_dir, with_homography=False)
    if match.n_frames != len(tracking.frame) or not np.array_equal(match.team, tracking.slot_team):
        raise ValueError('rule-based method and tracking inputs do not align')
    f = dead_ball.features(match)
    dead = dead_ball.detect_dead(match, dead_ball.dead_evidence(match, f), f)
    labels = model.build_labels(match, f, dead)[0]
    apply_published_dead_intervals(labels, tracking.blocks, config.get('published_dead_intervals', True))
    paths = games.asset_paths(data_dir=data_dir)
    return labels, {name: Path(paths[name]) for name in ('atd_meta', 'atd_raw', 'smart_tagging')}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate a tactical possession CSV; rules + TCN v6 is the default.')
    parser.add_argument("--config", default="configs/rules.yaml")
    parser.add_argument("--data-dir", default=None, help="asset root (default: MTAH_DATA_DIR or games.DATA_DIR)")
    parser.add_argument("--output", default="evaluation/results/Trivote.csv")
    parser.add_argument("--manifest", default=None, help="private manifest path (default: data/runs/<output>.manifest.json)")
    parser.add_argument('--method', choices=('rule_based_tcn', 'rule_based', 'old_rule_based'), default='rule_based_tcn',
                        help='rule_based_tcn: kural tabanlı yöntem + TCN; rule_based: kural tabanlı yöntem; old_rule_based: previous rules')
    parser.add_argument('--tcn-model', choices=tuple(MODELS), default='v6', help='TCN checkpoint for rule_based_tcn')
    parser.add_argument('--checkpoint', default=None, help='override checkpoint; requires --checkpoint-sha256')
    parser.add_argument('--checkpoint-sha256', default=None, help='SHA256 for an explicitly supplied checkpoint')
    args = parser.parse_args(argv)
    if bool(args.checkpoint) != bool(args.checkpoint_sha256):
        parser.error('--checkpoint and --checkpoint-sha256 must be supplied together')
    if args.method != 'rule_based_tcn' and (args.checkpoint or args.tcn_model != 'v6'):
        parser.error('TCN checkpoint options require --method rule_based_tcn')
    config_path = Path(args.config).resolve(); output = Path(args.output).resolve()
    config = load_config(config_path)
    if args.method == 'rule_based_tcn':
        checkpoint, digest = MODELS[args.tcn_model]
        checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else ROOT / checkpoint
        digest = args.checkpoint_sha256 or digest
        frozen_config = ROOT / 'configs/rule_based_tcn_frozen.json'
        frozen = json.loads(frozen_config.read_text(encoding='utf-8'))
        if text_sha256(config_path) != frozen['source_sha256']['configs/rules.yaml']:
            raise ValueError('config differs from the frozen rule-based method + TCN experiment')
        runs = ROOT / 'data/runs'; runs.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='rule_based_tcn_', dir=runs) as temp:
            temporary = Path(temp) / 'prediction.csv'
            cmd = [sys.executable, str(ROOT / 'scripts/predict_rule_based_tcn.py'), '--checkpoint', str(checkpoint),
                   '--expected-checkpoint-sha256', digest, '--frozen-config', str(frozen_config),
                   '--config', str(config_path), '--output', str(temporary)]
            if args.data_dir is not None: cmd += ['--data-dir', str(Path(args.data_dir).resolve())]
            subprocess.run(cmd, cwd=ROOT, check=True)
            manifest = json.loads(temporary.with_suffix('.manifest.json').read_text(encoding='utf-8'))
            output.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(temporary, output)
        manifest['output']['path'] = str(output)
        if manifest['output']['sha256'] != sha256(output): raise RuntimeError('copied CSV hash mismatch')
        manifest['method'] = 'rule_based_tcn'
        manifest['tcn_model'] = args.tcn_model if not args.checkpoint else 'custom'
    else:
        labels, inputs = (run_rule_based(config, args.data_dir) if args.method == 'rule_based'
                          else run_pipeline(config, args.data_dir))
        write_submission(output, labels)
        manifest = make_manifest(config_path, inputs, output)
        manifest['method'] = args.method
    path = Path(args.manifest) if args.manifest else Path("data/runs") / (Path(args.output).stem + ".manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"Submission written: {args.output}; manifest: {path}")


if __name__ == "__main__":
    main()
