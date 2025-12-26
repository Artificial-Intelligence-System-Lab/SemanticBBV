#!/usr/bin/env python3
"""
transform_bbtracker_gz.py

Directly read the .gz file output by Rust BbTracker, merge bb_id as needed and write back to new .gz,
no need to use JSON as intermediate format.

Usage example:

# Just copy without merging
python3 transform_bbtracker_gz.py \
    --input input.gz \
    --output output.gz

# Merge IDs (map.json content example: {"0": 10, "1": 10, "2": 2})
python3 transform_bbtracker_gz.py \
    --input input.gz \
    --output merged.gz \
    --map map.json
"""
import argparse
import gzip
import json
import re
import sys
from typing import Dict
import pickle

def load_map(path: str) -> Dict[int,int]:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def transform(
    input_gz: str,
    output_gz: str,
    id_map: Dict[int,int]
) -> None:
    """
    Read input_gz, parse ":bb_id:count" pairs per line, merge by id_map,
    write results to output_gz, format matches original BbTracker exactly.
    """
    pattern = re.compile(r":(\d+):(\d+)")
    with gzip.open(input_gz, 'rt', encoding='utf-8', errors='replace') as inp, \
         gzip.open(output_gz, 'wt', encoding='utf-8') as outp:
        for lineno, line in enumerate(inp, 1):
            line = line.rstrip('\n')
            if not line:
                # Skip empty lines
                continue
            if not line.startswith('T'):
                sys.stderr.write(f"Warning: line {lineno} does not start with 'T', write as-is\n")
                outp.write(line + "\n")
                continue

            # Extract all (bb_id, count)
            pairs = pattern.findall(line)
            merged: Dict[int,int] = {}
            for bbid_s, cnt_s in pairs:
                old_id = int(bbid_s)
                count = int(cnt_s)
                new_id = id_map.get(old_id, old_id)
                merged[new_id] = merged.get(new_id, 0) + count

            # Rebuild a line: 'T' + for each sorted id => ":id:count   " + "\n"
            if merged:
                parts = [ f":{bid + 1}:{merged[bid]}   "
                          for bid in sorted(merged.keys()) ]
                outp.write("T" + "".join(parts).rstrip() + "\n")
            else:
                # If no bb, output "T\n"
                outp.write("T\n")

def main():
    parser = argparse.ArgumentParser(
        description="Directly merge and rewrite BbTracker .gz file, without JSON intermediate format"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input .gz file"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to output .gz file"
    )
    parser.add_argument(
        "--map", "-m",
        help="Optional pickle mapping file: {old_id: new_id, ...}"
    )
    args = parser.parse_args()

    id_map: Dict[int,int] = {}
    if args.map:
        try:
            id_map = load_map(args.map)
        except Exception as e:
            sys.stderr.write(f"Error: unable to read map file {args.map}: {e}\n")
            sys.exit(1)

    try:
        transform(args.input, args.output, id_map)
    except Exception as e:
        sys.stderr.write(f"Error during transform: {e}\n")
        sys.exit(1)

    print(f"Done. Output written to {args.output}", file=sys.stderr)

if __name__ == "__main__":
    main()