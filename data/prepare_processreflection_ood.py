import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / ".hf_cache"))

from data.load_dataset import load_dataset_hf
from data.preprocess import run_proprocessing
from data.split_tasks import split_tasks


DATASETS_ROOT = REPO_ROOT / "datasets"
TEST_RATIO = 0.1
SEED = 42

TARGET_SPECS = {
    "biology": {
        "path": DATASETS_ROOT / "sciknoweval-biology",
        "kind": "sciknoweval",
        "source_dataset": "Biology",
        "legacy_dir": DATASETS_ROOT / "sciknoweval" / "biology",
    },
    "chemistry": {
        "path": DATASETS_ROOT / "sciknoweval-chemistry",
        "kind": "sciknoweval",
        "source_dataset": "Chemistry",
        "legacy_dir": DATASETS_ROOT / "sciknoweval" / "chemistry",
    },
    "material": {
        "path": DATASETS_ROOT / "sciknoweval-material",
        "kind": "sciknoweval",
        "source_dataset": "Material",
        "legacy_dir": DATASETS_ROOT / "sciknoweval" / "material",
    },
    "physics": {
        "path": DATASETS_ROOT / "sciknoweval-physics",
        "kind": "sciknoweval",
        "source_dataset": "Physics",
        "legacy_dir": DATASETS_ROOT / "sciknoweval" / "physics",
    },
    "tooluse": {
        "path": DATASETS_ROOT / "tooluse",
        "kind": "tooluse",
    },
    "aime24": {
        "path": DATASETS_ROOT / "math-aime24",
        "kind": "math",
        "source_dataset": "math-ai/aime24",
    },
    "aime25": {
        "path": DATASETS_ROOT / "math-aime25",
        "kind": "math",
        "source_dataset": "math-ai/aime25",
    },
}


def _artifact_paths(target_dir: Path) -> dict[str, Path]:
    return {
        "train_json": target_dir / "train.json",
        "test_json": target_dir / "test.json",
        "train_parquet": target_dir / "train.parquet",
        "test_parquet": target_dir / "test.parquet",
    }


def _copy_if_exists(src: Path, dst: Path, force: bool) -> None:
    if not src.exists():
        return
    if dst.exists() and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_legacy_split(legacy_dir: Path, target_dir: Path, force: bool) -> None:
    for name in ("train.json", "test.json", "train.parquet", "test.parquet"):
        _copy_if_exists(legacy_dir / name, target_dir / name, force=force)


def _ensure_sciknoweval(spec: dict, force: bool) -> None:
    target_dir = spec["path"]
    artifacts = _artifact_paths(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    legacy_dir = spec.get("legacy_dir")
    if legacy_dir is not None:
        _copy_legacy_split(legacy_dir, target_dir, force=force)

    if artifacts["train_json"].exists() and artifacts["test_json"].exists() and not force:
        return

    raw_json = target_dir / f"{spec['source_dataset'].lower()}.json"
    load_dataset_hf(spec["source_dataset"], str(raw_json))
    split_tasks(str(raw_json), str(target_dir), test_ratio=TEST_RATIO, seed=SEED)


def _ensure_math(spec: dict, force: bool) -> None:
    target_dir = spec["path"]
    artifacts = _artifact_paths(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if artifacts["train_json"].exists() and artifacts["test_json"].exists() and not force:
        return

    raw_json = target_dir / f"{spec['path'].name}.json"
    load_dataset_hf(spec["source_dataset"], str(raw_json))
    split_tasks(str(raw_json), str(target_dir), test_ratio=TEST_RATIO, seed=SEED)


def _ensure_tooluse(spec: dict) -> None:
    spec["path"].mkdir(parents=True, exist_ok=True)


def _ensure_parquet(spec: dict, force: bool) -> None:
    target_dir = spec["path"]
    artifacts = _artifact_paths(target_dir)
    if (
        artifacts["train_parquet"].exists()
        and artifacts["test_parquet"].exists()
        and not force
    ):
        return
    run_proprocessing(str(target_dir))


def prepare_target(target: str, force: bool) -> None:
    if target not in TARGET_SPECS:
        raise ValueError(f"Unknown target: {target}")

    spec = TARGET_SPECS[target]
    if spec["kind"] == "sciknoweval":
        _ensure_sciknoweval(spec, force=force)
    elif spec["kind"] == "math":
        _ensure_math(spec, force=force)
    elif spec["kind"] == "tooluse":
        _ensure_tooluse(spec)
    else:
        raise ValueError(f"Unsupported target kind: {spec['kind']}")

    _ensure_parquet(spec, force=force)


def parse_targets(raw_targets: list[str]) -> list[str]:
    targets: list[str] = []
    for value in raw_targets:
        for target in value.split(","):
            target = target.strip()
            if target:
                targets.append(target)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare flat dataset directories for ProcessReflection-OOD.")
    parser.add_argument("--targets", nargs="+", required=True, help="Comma-separated or space-separated target names.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing artifacts.")
    args = parser.parse_args()

    for target in parse_targets(args.targets):
        print(f"Preparing target: {target}")
        prepare_target(target, force=args.force)


if __name__ == "__main__":
    main()
