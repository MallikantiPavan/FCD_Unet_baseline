from __future__ import annotations

import argparse
from collections import Counter

from dataset import load_subject_dataframe, split_dataframe
from utils.config import load_config


def _count_groups(frame) -> Counter[str]:
	if "group" not in frame.columns:
		raise ValueError("CSV must contain a group column with values such as 'hc' and 'fcd'")
	return Counter(str(value).strip().lower() for value in frame["group"])


def _print_counts(name: str, frame) -> None:
	counts = _count_groups(frame)
	print(
		f"{name}: total={len(frame)}, "
		f"healthy_controls={counts.get('hc', 0)}, "
		f"fcd_patients={counts.get('fcd', 0)}"
	)
	unexpected = sorted(set(counts) - {"hc", "fcd"})
	if unexpected:
		print(f"  Unexpected group labels: {unexpected}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Count healthy controls and FCD patients in each split")
	parser.add_argument("--config", default="config/config.yaml")
	args = parser.parse_args()

	config = load_config(args.config)
	train_frame = load_subject_dataframe(config["data"]["train_csv"])
	test_frame = load_subject_dataframe(config["data"]["test_csv"])
	train_split, validation_split = split_dataframe(
		train_frame,
		float(config["split"]["train_ratio"]),
		int(config["seed"]),
	)

	_print_counts("Train", train_split)
	_print_counts("Validation", validation_split)
	_print_counts("Test", test_frame)


if __name__ == "__main__":
	main()
