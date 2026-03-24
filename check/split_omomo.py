#!/usr/bin/env python3
"""Split OMOMO .pt files into folders by object name.

Given files named like `sub{num}_{object_name}_{NNN}.pt` in a source directory
(e.g. `OMOMO_new_all`), this script moves each file into a directory named
`OMOMO_new_{object_name}` placed next to the source directory.

Usage:
  python check/split_omomo.py --src /home/rllab/haechan/holosoma_prev/src/holosoma_retargeting/holosoma_retargeting/demo_data/OMOMO_new_all/ --dry-run

By default the script moves files; pass `--dry-run` to only preview actions.
"""

import argparse
import os
import re
import shutil
from collections import defaultdict


DEFAULT_SRC = "/home/rllab/haechan/holosoma_prev/src/holosoma_retargeting/holosoma_retargeting/demo_data/OMOMO_new_all/"

RE_PATTERN = re.compile(r"^sub\d+_(.+)_(\d{3})\.pt$")


def find_pt_files(src_dir):
	for entry in os.listdir(src_dir):
		if not entry.lower().endswith('.pt'):
			continue
		yield os.path.join(src_dir, entry), entry


def object_name_from_filename(fname: str) -> str | None:
	m = RE_PATTERN.match(fname)
	if not m:
		return None
	return m.group(1)


def main():
	p = argparse.ArgumentParser(description='Split OMOMO .pt files into object folders')
	p.add_argument('--src', '-s', default=DEFAULT_SRC, help='Source directory containing .pt files')
	p.add_argument('--dry-run', action='store_true', help='Print actions without moving files')
	p.add_argument('--verbose', '-v', action='store_true')
	args = p.parse_args()

	src = os.path.abspath(args.src)
	if not os.path.isdir(src):
		print('Source directory not found:', src)
		return

	parent = os.path.dirname(src.rstrip(os.sep))

	moved_counts = defaultdict(int)
	skipped = []

	for fullpath, filename in find_pt_files(src):
		obj = object_name_from_filename(filename)
		if not obj:
			skipped.append(filename)
			continue

		dest_dir = os.path.join(parent, f'OMOMO_new_{obj}')
		os.makedirs(dest_dir, exist_ok=True)
		dest_path = os.path.join(dest_dir, filename)

		if args.dry_run or args.verbose:
			print(f"{('DRY:' if args.dry_run else 'MOVE:')} {fullpath} -> {dest_path}")

		if not args.dry_run:
			try:
				shutil.move(fullpath, dest_path)
				moved_counts[obj] += 1
			except Exception as e:
				print('Failed to move', fullpath, '->', dest_path, 'error:', e)

	print('\nSummary:')
	total = sum(moved_counts.values())
	print('  total moved files:', total)
	for obj, c in sorted(moved_counts.items()):
		print(f'  {obj}: {c}')
	if skipped:
		print('\nSkipped files (did not match pattern):')
		for s in skipped:
			print(' ', s)


if __name__ == '__main__':
	main()
