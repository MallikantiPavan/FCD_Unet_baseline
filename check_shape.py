from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import nibabel as nib


def summarize_shapes(data_dir: str | Path, pattern: str) -> tuple[Counter, int, list[str]]:
    """Return counts grouped by actual NIfTI shape."""
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    shapes = Counter()
    checked = 0
    mismatches = []
    for path in sorted(root.glob(f"*/{pattern}")):
        checked += 1
        try:
            shape = tuple(nib.load(str(path)).shape)
        except Exception as error:
            mismatches.append(f"{path}: could not read file ({error})")
            continue
        shapes[shape] += 1
    return shapes, checked, mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description="Count NIfTI files with a specific volume shape")
    parser.add_argument("--data-dir", help="Root directory containing subject folders",default="/storage/projects/vinkle/ez_compass_imaging/data/cropped_data_pretty")
    parser.add_argument("--pattern", default="*.nii.gz", help="Filename pattern, for example T1w_brain.nii.gz")
    parser.add_argument("--show-errors", action="store_true", help="Print files that could not be read")
    args = parser.parse_args()

    shapes, checked, errors = summarize_shapes(args.data_dir, args.pattern)
    print(f"Pattern: {args.pattern}")
    print(f"Files checked: {checked}")
    print("\nShape summary:")
    for shape, count in sorted(shapes.items()):
        print(f"{shape}: {count} file(s)")
    print(f"Unreadable files: {len(errors)}")
    if args.show_errors and errors:
        print("\nUnreadable files:")
        print("\n".join(errors))


if __name__ == "__main__":
    main()