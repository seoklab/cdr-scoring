#!/usr/bin/env python3
"""Check GalaxyRefine pertMD outputs for both fix_type modes.

Targets skipped during input preparation are excluded when they are absent from
both output roots. A target is considered successful for a mode when
input/model/model.pdb exists under that mode's target directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence


BASE_DIR = Path("/home/sujin/DB/h3-loop-modeling/ab_ag/1_tr_abag")
DEFAULT_SOURCE_DIR = BASE_DIR / "00_crystal"
DEFAULT_OUTPUT_DIRS = {
    "all": BASE_DIR / "41_pertmd_cdr_input",
    "none": BASE_DIR / "42_pertmd_all_input",
}
DEFAULT_EXPECTED_SKIPPED = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--fix-all-dir", type=Path, default=DEFAULT_OUTPUT_DIRS["all"],
                        help="Output root for fix_type=all.")
    parser.add_argument("--fix-none-dir", type=Path, default=DEFAULT_OUTPUT_DIRS["none"],
                        help="Output root for fix_type=none.")
    parser.add_argument("--target", action="append", default=None,
                        help="Target ID without .pdb. Can be passed multiple times.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--expected-skipped", type=int, default=DEFAULT_EXPECTED_SKIPPED,
                        help="Expected number of targets absent from both output roots.")
    parser.add_argument("--show-ok", action="store_true",
                        help="Print targets that have model.pdb for both modes.")
    return parser.parse_args()


def target_ids(source_dir: Path, requested: Sequence[str] | None, limit: int | None) -> List[str]:
    if requested:
        targets = list(requested)
    else:
        targets = sorted(path.stem for path in source_dir.glob("*.pdb"))
    return targets[:limit] if limit is not None else targets


def model_path(output_dir: Path, target: str) -> Path:
    return output_dir / target / "input" / "model" / "model.pdb"


def target_dir_exists(output_dir: Path, target: str) -> bool:
    return (output_dir / target).is_dir()


def main() -> int:
    args = parse_args()
    output_dirs: Dict[str, Path] = {
        "all": args.fix_all_dir,
        "none": args.fix_none_dir,
    }

    targets = target_ids(args.source_dir, args.target, args.limit)
    skipped: List[str] = []
    failures: List[str] = []
    ok: List[str] = []

    for target in targets:
        present = {
            fix_type: target_dir_exists(output_dir, target)
            for fix_type, output_dir in output_dirs.items()
        }
        if not any(present.values()):
            skipped.append(target)
            continue

        missing = [
            fix_type
            for fix_type, output_dir in output_dirs.items()
            if not model_path(output_dir, target).is_file()
        ]
        if missing:
            details = ", ".join(
                f"{fix_type}:{model_path(output_dirs[fix_type], target)}"
                for fix_type in missing
            )
            failures.append(f"{target}\tmissing model.pdb\t{details}")
        else:
            ok.append(target)

    print(f"source_targets={len(targets)}")
    print(f"skipped_absent_from_both_outputs={len(skipped)}")
    print(f"checked_targets={len(targets) - len(skipped)}")
    print(f"ok_targets={len(ok)}")
    print(f"failed_targets={len(failures)}")

    status = 0
    if len(skipped) != args.expected_skipped:
        status = 2
        print(
            f"ERROR expected_skipped={args.expected_skipped}, "
            f"observed_skipped={len(skipped)}"
        )

    if skipped:
        print("Skipped targets:")
        for target in skipped:
            print(f"  {target}")

    if failures:
        status = 1
        print("Failed targets:")
        for failure in failures:
            print(f"  {failure}")

    if args.show_ok and ok:
        print("OK targets:")
        for target in ok:
            print(f"  {target}")

    return status


if __name__ == "__main__":
    raise SystemExit(main())
