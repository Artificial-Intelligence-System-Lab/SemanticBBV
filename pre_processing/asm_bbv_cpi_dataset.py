#!/usr/bin/env python3
"""
transform_bbtracker_gz.py

Directly read the .gz file output by Rust BbTracker, merge bb_id as needed and write out the pickle (triplet) file required for triplet-loss training, and support optional semi-hard negative sample mining.

By default, sample up to 10000 triplets, can be modified via --max-triplets.

Usage example:

# 1. Use recommended hybrid strategy (hybrid-knn), k=20, enable semi-hard
python3 transform_bbtracker_gz.py \\
    --input input.gz \\
    --strategy hybrid-knn \\
    --positive-k 20 \\
    --semi-hard

# 2. Use pure K-nearest neighbor strategy
python3 transform_bbtracker_gz.py \\
    --input input.gz \\
    --strategy knn \\
    --positive-k 5

# 3. Use old statistical strategy
python3 transform_bbtracker_gz.py \\
    --input input.gz \\
    --strategy statistical
"""
import argparse
import gzip
import os
import pickle
import random
import re
import sys
import traceback
import json
from typing import Dict, List, Tuple, Literal


import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

# Fix random seed to ensure reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_map(path: str) -> Dict[int, np.ndarray]:
    with open(path, 'rb') as f:
        return pickle.load(f)

# --- Old build function (statistical) ---
def build_triplets(sim: np.ndarray, bbvs: List[Dict[int,int]], cpi_array: List[float], pos_thresh: float, neg_thresh: float, max_triplets: int) -> List[Tuple]:
    n = sim.shape[0]
    neg_cands = {i: [k for k in range(n) if k != i and sim[i,k] < neg_thresh] for i in range(n)}
    pos_pairs = [(i, j) for i in range(n) for j in range(i+1, n) if sim[i,j] > pos_thresh]
    random.shuffle(pos_pairs)
    triplets = []
    for a, p in pos_pairs:
        if len(triplets) >= max_triplets: break
        cands = neg_cands.get(a, [])
        if not cands: continue
        n_idx = random.choice(cands)
        triplets.append((bbvs[a], bbvs[p], bbvs[n_idx], cpi_array[a], cpi_array[p], cpi_array[n_idx]))
    print(f"[Triplets] Simple sampling constructed {len(triplets)} triplets in total (limit {max_triplets})")
    return triplets

def build_semi_hard_triplets(sim: np.ndarray, bbvs: List[Dict[int,int]], cpi_array: List[float], pos_thresh: float, margin: float, max_triplets: int) -> List[Tuple]:
    n = sim.shape[0]
    pos_pairs = [(i, j) for i in range(n) for j in range(i+1, n) if sim[i,j] > pos_thresh]
    random.shuffle(pos_pairs)
    triplets = []
    for a, p in pos_pairs:
        if len(triplets) >= max_triplets: break
        sp = sim[a,p]
        negs = [k for k in range(n) if k != a and k != p and sim[a,k] < sp and sim[a,k] > sp - margin]
        if not negs: continue
        n_idx = random.choice(negs)
        triplets.append((bbvs[a], bbvs[p], bbvs[n_idx], cpi_array[a], cpi_array[p], cpi_array[n_idx]))
    print(f"[Triplets] Semi-hard sampling constructed {len(triplets)} triplets in total (limit {max_triplets})")
    return triplets

# --- Pure KNN build function ---
def build_triplets_knn(sim: np.ndarray, bbvs: List[Dict[int,int]], cpi_array: List[float], positive_k: int, use_semi_hard: bool, margin: float, max_triplets: int) -> List[Tuple]:
    n = sim.shape[0]
    if n < positive_k + 2:
        print(f"Warning: Sample count ({n}) is too small, cannot use k={positive_k} for KNN sampling.")
        return []
    sorted_indices = np.argsort(sim, axis=1)
    pos_neighbors = {i: sorted_indices[i, -(positive_k + 1):-1][::-1].tolist() for i in range(n)}
    neg_candidates = {i: sorted_indices[i, :-(positive_k + 1)].tolist() for i in range(n)}
    triplets = []
    anchor_indices = list(range(n))
    random.shuffle(anchor_indices)
    for a_idx in anchor_indices:
        if len(triplets) >= max_triplets: break
        pos_pool = pos_neighbors.get(a_idx, [])
        if not pos_pool: continue
        p_idx = random.choice(pos_pool)
        neg_pool = neg_candidates.get(a_idx, [])
        if not neg_pool: continue
        n_idx = -1
        if use_semi_hard:
            sim_ap = sim[a_idx, p_idx]
            semi_hard_negs = [k for k in neg_pool if sim[a_idx, k] < sim_ap and sim[a_idx, k] > sim_ap - margin]
            if semi_hard_negs: n_idx = random.choice(semi_hard_negs)
        if n_idx == -1: n_idx = random.choice(neg_pool)
        triplets.append((bbvs[a_idx], bbvs[p_idx], bbvs[n_idx], cpi_array[a_idx], cpi_array[p_idx], cpi_array[n_idx]))
    print(f"[Triplets] Pure-KNN (k={positive_k}, semi-hard={use_semi_hard}) sampling constructed {len(triplets)} triplets in total")
    return triplets

# --- Hybrid strategy build function (Hybrid-KNN) ---
def build_triplets_hybrid_knn(sim: np.ndarray, bbvs: List[Dict[int,int]], cpi_array: List[float], positive_k: int, pos_thresh: float, use_semi_hard: bool, margin: float, max_triplets: int) -> List[Tuple]:
    n = sim.shape[0]
    if n < positive_k + 2:
        print(f"Warning: Sample count ({n}) is too small, cannot use k={positive_k} for Hybrid-KNN sampling.")
        return []
    sorted_indices = np.argsort(sim, axis=1)
    pos_neighbors = {i: sorted_indices[i, -(positive_k + 1):-1][::-1].tolist() for i in range(n)}
    neg_candidates = {i: sorted_indices[i, :-(positive_k + 1)].tolist() for i in range(n)}
    triplets = []
    anchor_indices = list(range(n))
    random.shuffle(anchor_indices)
    for a_idx in anchor_indices:
        if len(triplets) >= max_triplets: break
        pos_pool = pos_neighbors.get(a_idx, [])
        if not pos_pool: continue
        # --- Core change: double filtering ---
        qualified_pos_pool = [p for p in pos_pool if sim[a_idx, p] > pos_thresh]
        if not qualified_pos_pool: continue
        p_idx = random.choice(qualified_pos_pool)
        neg_pool = neg_candidates.get(a_idx, [])
        if not neg_pool: continue
        n_idx = -1
        if use_semi_hard:
            sim_ap = sim[a_idx, p_idx]
            semi_hard_negs = [k for k in neg_pool if sim[a_idx, k] < sim_ap and sim[a_idx, k] > sim_ap - margin]
            if semi_hard_negs: n_idx = random.choice(semi_hard_negs)
        if n_idx == -1: n_idx = random.choice(neg_pool)
        triplets.append((bbvs[a_idx], bbvs[p_idx], bbvs[n_idx], cpi_array[a_idx], cpi_array[p_idx], cpi_array[n_idx]))
    print(f"[Triplets] Hybrid-KNN (k={positive_k}, thresh>{pos_thresh:.4f}, semi-hard={use_semi_hard}) sampling constructed {len(triplets)} triplets in total")
    return triplets

def similarity_matching(input_gz: str, cpi_array: List[float], strategy: Literal['statistical', 'knn', 'hybrid-knn'], positive_k: int, use_semi_hard: bool, margin: float, max_triplets: int) -> List[Tuple]:
    # 1. Read bbvs
    pat = re.compile(r":(\d+):(\d+)")
    bbvs = []
    with gzip.open(input_gz, 'rt', encoding='utf-8', errors='replace') as inp:
        for line in inp:
            line = line.strip()
            if line.startswith('T'):
                d = {int(k): int(v) for k, v in pat.findall(line)}
                bbvs.append(d)
    n = len(bbvs)
    if n < 3:
        print("Too few samples, cannot construct triplets", file=sys.stderr)
        sys.exit(1)
    
    # Ensure bbvs and cpi_array have the same length
    cpi_length = len(cpi_array)
    if cpi_length > n: cpi_array = cpi_array[:n]
    elif cpi_length < n: bbvs = bbvs[:cpi_length]
    n = len(bbvs)

    # 2. Random projection
    max_id = max(k for d in bbvs for k in d.keys()) if any(bbvs) else 0
    proj = np.random.randn(max_id + 1, 100)
    vectors = np.vstack([np.array([d.get(i, 0) for i in range(max_id + 1)]).dot(proj) for d in bbvs])

    # 3. Cosine similarity
    sim = cosine_similarity(vectors)

    # 4 & 5. Construct triplets based on strategy
    dataset = []
    if strategy == 'knn':
        print(f"[Strategy] Using Pure K-Nearest Neighbors (k={positive_k}).")
        dataset = build_triplets_knn(sim, bbvs, cpi_array, positive_k, use_semi_hard, margin, max_triplets)
    else: # 'statistical' or 'hybrid-knn' both need to calculate global thresholds
        sims = sim[~np.eye(n, dtype=bool)]
        mu, sigma = sims.mean(), sims.std()
        pos_thresh = mu + 0.5 * sigma
        neg_thresh = mu - 1.0 * sigma
        
        if strategy == 'hybrid-knn':
            print(f"[Strategy] Using Hybrid-KNN (k={positive_k}, global_thresh>{pos_thresh:.4f}).")
            dataset = build_triplets_hybrid_knn(sim, bbvs, cpi_array, positive_k, pos_thresh, use_semi_hard, margin, max_triplets)
        else: # 'statistical'
            print(f"[Strategy] Using Statistical thresholds (pos>{pos_thresh:.4f}, neg<{neg_thresh:.4f}).")
            if use_semi_hard:
                dataset = build_semi_hard_triplets(sim, bbvs, cpi_array, pos_thresh, margin, max_triplets)
            else:
                dataset = build_triplets(sim, bbvs, cpi_array, pos_thresh, neg_thresh, max_triplets)
    
    return dataset


def transform(dataset: List[Tuple], output: str, id_map: Dict[int, np.ndarray]) -> None:
    new_data = []
    if not id_map:
        print("Warning: id_map is empty, cannot transform data.")
        return
    print(f"[Transform] id_map key range: {min(id_map)}–{max(id_map)}")
    for a, p, n, a_cpi, p_cpi, n_cpi in dataset:
        def to_emb_w(d: Dict[int,int]):
            if not d: return [], []
            items = sorted(d.items(), key=lambda kv: kv[1])
            keys, ws = zip(*items)
            embs = [id_map[k-1] for k in keys if k-1 in id_map]
            valid_ws = [w for k, w in zip(keys, ws) if k-1 in id_map]
            return embs, valid_ws
        ae, aw = to_emb_w(a)
        pe, pw = to_emb_w(p)
        ne, nw = to_emb_w(n)
        if ae and pe and ne: # Ensure all triplets are valid
            new_data.append((ae, aw, pe, pw, ne, nw, a_cpi, p_cpi, n_cpi))
    with open(output, 'wb') as f:
        pickle.dump(new_data, f)
    print(f"[Transform] Wrote {len(new_data)} records to {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--input',  '-i', required=True, help='Input .gz file')
    p.add_argument('--map',    '-m', help='bb_id→emb pickle file (default: <input>.vectors.pkl)')
    p.add_argument('--cpi',    '-c', help='simulation results file (default: <input>_timeline_full.json)')
    p.add_argument('--output', '-o', help='Output triplets pickle file (default: <input>.emb.pkl)')
    p.add_argument('--max-triplets', type=int, default=10000, help='Maximum number of triplets to sample per file (default 10000)')
    
    p.add_argument('--strategy', choices=['statistical', 'knn', 'hybrid-knn'], default='hybrid-knn',
                   help='Choose triplet mining strategy:\n'
                        '  statistical: (old) Set thresholds based on global mean/std.\n'
                        '  knn:         (new) Define positive/negative samples based on local K-nearest neighbors.\n'
                        '  hybrid-knn:  (recommended) Combine knn and statistical, positive samples must satisfy both\n'
                        '               local neighbor and global similarity dual criteria.\n'
                        '(default: %(default)s)')
    p.add_argument('--positive-k', type=int, default=20,
                   help='When strategy=knn or hybrid-knn, define the K value for K-nearest neighbors.\n'
                        'For hybrid-knn, it is recommended to set a slightly larger value. (default: %(default)s)')
    p.add_argument('--semi-hard', action='store_true', help='Enable semi-hard negative sample mining.')
    p.add_argument('--margin', type=float, default=0.25, help='Margin for semi-hard mining. (default: %(default)s)')
    
    args = p.parse_args()

    base, _ = os.path.splitext(args.input)
    print(f"Processing: {base}")
    args.map = args.map or base + ".vectors.pkl"
    args.output = args.output or base + ".emb.pkl"
    args.cpi = args.cpi or base + "_timeline_full.json"
    
    if not os.path.exists(args.cpi):
        sys.stderr.write(f"Error: Unable to load stats file {args.cpi}\n"); sys.exit(1)
    
    cpi_array = []
    with open(args.cpi, "r") as cpi_file:
        cpi_data = json.load(cpi_file)
    if "board.processor.start.core.numCycles" not in cpi_data or "simInsts" not in cpi_data:
        sys.stderr.write(f"Error: CPI file {args.cpi} is missing required keys.\n"); sys.exit(1)
    
    sim_insts_diff = np.diff(cpi_data["simInsts"])
    num_cycles_list = cpi_data["board.processor.start.core.numCycles"]
    num_cycles = np.array(num_cycles_list) 
    num_cycles_sliced = num_cycles[1:len(sim_insts_diff) + 1]
    
    # Use numpy for efficient computation and filtering
    valid_mask = sim_insts_diff > 0
    cpi_values = num_cycles_sliced[valid_mask] / sim_insts_diff[valid_mask] 
    cpi_array = cpi_values.tolist()
    initial_cpi = cpi_data["board.processor.start.core.numCycles"][0] / cpi_data["simInsts"][0]
    cpi_array.insert(0, initial_cpi)

    try: id_map = load_map(args.map)
    except Exception as e:
        sys.stderr.write(f"Error: Unable to load map file {args.map}: {e}\n"); traceback.print_exc(); sys.exit(1)

    try:
        triplets = similarity_matching(
            args.input, cpi_array,
            strategy=args.strategy,
            positive_k=args.positive_k,
            use_semi_hard=args.semi_hard,
            margin=args.margin,
            max_triplets=args.max_triplets
        )
        if triplets:
            print(f"[Main] Successfully constructed {len(triplets)} triplets (limit {args.max_triplets})")
            transform(triplets, args.output, id_map)
        else:
            print("[Main] Failed to construct any triplets. Please check input data or parameters.")
    except Exception as e:
        sys.stderr.write(f"Error: An error occurred during processing: {e}\n"); traceback.print_exc(); sys.exit(1)


if __name__ == "__main__":
    main()