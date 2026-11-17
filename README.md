# Trivote — More than a Hack 2027, Phase 1: Ball Possession

This is team **Trivote**'s submission for [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027).
It labels every frame of the Barcelona–Athletic tactical video as `H` (home possession), `A` (away
possession), `D` (dead ball) or `X` (not graded).

The method is a rule-based possession and dead-ball system built on the Laws of the Game. A small
temporal convolutional network (TCN v6), trained only on other matches, adds bounded evidence to it.
The full description is in **[docs/Trivote.md](docs/Trivote.md)**: approach, assets used, training
data, possession rules, limitations and reproduction.

## Reproduce the submission

1. Install [uv](https://docs.astral.sh/uv/). It provides Python 3.11 and every pinned dependency
   from `uv.lock`.
2. Unpack the organisers' match bundle into `data/` (see [DATA.md](DATA.md)). Only the files under
   `data/tactical/` are read. The match data is not redistributable, so it is not in this
   repository.
3. Run:

```bash
uv sync --locked --extra starter --extra training
uv run --locked --extra starter --extra training python scripts/rule_based/pipeline.py
uv run --locked --extra starter python evaluation/validate_submission.py data/runs/final_pipeline/Trivote.csv
```

Inference runs on the CPU and takes under a minute. It writes:
- `data/runs/final_pipeline/Trivote.csv`, the possession file;
- `data/runs/Trivote.manifest.json`, a manifest with the hashes of inputs, config, checkpoint and
  source;
- a VLC `.ass` subtitle overlay next to the tactical video.

The expected SHA256 of `Trivote.csv` is:

```
3cf89a2c375dcfc6eedd1a4ed5489c9726482d9f65c0f2cfdc17ee98fa91f7f9
```

To check determinism, run inference in two clean processes and compare the hashes:

```bash
uv run --locked --extra starter --extra training python scripts/check_reproduction.py \
    --run-dir data/runs/final_pipeline/reproduction \
    --final-output data/runs/final_pipeline/reproduction/Trivote.csv
```

The trained checkpoint `data/runs/training/dfl_v6.pt` is included. Inference refuses any checkpoint
whose SHA256 does not match. Retraining commands are in [docs/Trivote.md](docs/Trivote.md#6-reproduction).

## Repository layout

| Path | Contents |
|---|---|
| `src/possession/rule_based/` | Dead-ball detector, possession decoder, ATD loader |
| `src/possession/` | Feature extraction, TCN model and training, inference entry point (`predict.py`) |
| `scripts/rule_based/pipeline.py` | One-command inference and overlay |
| `scripts/` | Inference, reproduction check, external-data fetching, training and research evaluations |
| `configs/` | Frozen rule config and the frozen rules + TCN fusion protocol |
| `data/runs/training/` | TCN checkpoints (v6 default, v5 optional) |
| `evaluation/`, `games.py` | Organisers' scorer, validator and match helpers, unchanged |
| `starter/` | Organisers' starter loaders and notebooks |
| `tests/` | Unit tests: `uv run --locked --extra starter --extra training python -m unittest discover -s tests -p "test_*.py"` |

The challenge description and rules are kept as published by the organisers: [CHALLENGE.md](CHALLENGE.md),
[RULES.md](RULES.md), [EVALUATION.md](EVALUATION.md), [SUBMISSION.md](SUBMISSION.md),
[DATA.md](DATA.md), [FAQ.md](FAQ.md).

## Data and licences

- The TCN was trained on the IDSSE/DFL Bundesliga dataset (Bassek et al., *Scientific Data* 2025,
  CC BY 4.0) and on Metrica Sports sample data
  ([metrica-sports/sample-data](https://github.com/metrica-sports/sample-data); attribution to
  Metrica Sports). No labels of the provided match were used.
- Code and documentation are under the MIT licence in [LICENSE](LICENSE). The licence covers this
  repository's contents only, not the gated match data.
