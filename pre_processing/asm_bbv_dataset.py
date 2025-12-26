#!/usr/bin/env python3
"""
transform_bbtracker_gz.py

Directly read the .gz file output by Rust BbTracker, merge bb_id as needed and write out the pickle (triplet) file required for triplet-loss training, and support optional semi-hard negative sample mining.

By default, sample up to 10000 triplets, can be modified via --max-triplets.

Usage example:

# Only provide input, use default map/output, simple negative examples, up to 10000 triplets
python3 transform_bbtracker_gz.py --input input.gz

# Specify semi-hard, margin=0.2, up to 5000 triplets
python3 transform_bbtracker_gz.py \
    --input input.gz \
    --semi-hard \
    --margin 0.2 \
    --max-triplets 5000
"""
import argparse
import gzip
import os
import pickle
import random
import re
import sys
import traceback
from typing import Dict, List, Tuple

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


def plot_tsne_clusters(vectors: np.ndarray,
                       labels: np.ndarray,
                       clusters_id: List[List[int]],
                       save_path: str = None):
    tsne = TSNE(n_components=2, random_state=SEED)
    vec2d = tsne.fit_transform(vectors)

    n_clusters = len(clusters_id)
    colors = list(mcolors.TABLEAU_COLORS.values()) * (n_clusters // 10 + 1)

    plt.figure(figsize=(12, 8))
    for idx in range(n_clusters):
        mask = (labels == idx)
        pts = vec2d[mask]
        plt.scatter(pts[:, 0], pts[:, 1],
                    c=[colors[idx]],
                    label=f'Cluster {idx} ({len(clusters_id[idx])})',
                    alpha=0.7, s=30)
    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[t-SNE] saved to {save_path}")
    else:
        plt.show()
    plt.close()


def build_triplets(sim: np.ndarray,
                   bbvs: List[Dict[int,int]],
                   pos_thresh: float,
                   neg_thresh: float,
                   max_triplets: int
                  ) -> List[Tuple[Dict[int,int],Dict[int,int],Dict[int,int]]]:
    n = sim.shape[0]
    neg_cands = {
        i: [k for k in range(n) if k != i and sim[i,k] < neg_thresh]
        for i in range(n)
    }
    pos_pairs = [
        (i, j)
        for i in range(n) for j in range(i+1, n)
        if sim[i,j] > pos_thresh
    ]
    random.shuffle(pos_pairs)

    triplets = []
    for a, p in pos_pairs:
        if len(triplets) >= max_triplets:
            break
        cands = neg_cands.get(a, [])
        if not cands:
            continue
        n_idx = random.choice(cands)
        triplets.append((bbvs[a], bbvs[p], bbvs[n_idx]))

    print(f"[Triplets] Simple sampling constructed {len(triplets)} triplets in total (limit {max_triplets})")
    return triplets


def build_semi_hard_triplets(sim: np.ndarray,
                             bbvs: List[Dict[int,int]],
                             pos_thresh: float,
                             margin: float,
                             max_triplets: int
                            ) -> List[Tuple[Dict[int,int],Dict[int,int],Dict[int,int]]]:
    n = sim.shape[0]
    pos_pairs = [
        (i, j)
        for i in range(n) for j in range(i+1, n)
        if sim[i,j] > pos_thresh
    ]
    random.shuffle(pos_pairs)

    triplets = []
    for a, p in pos_pairs:
        if len(triplets) >= max_triplets:
            break
        sp = sim[a,p]
        negs = [
            k for k in range(n)
            if k != a and k != p
               and sim[a,k] < sp
               and sim[a,k] > sp - margin
        ]
        if not negs:
            continue
        n_idx = random.choice(negs)
        triplets.append((bbvs[a], bbvs[p], bbvs[n_idx]))

    print(f"[Triplets] Semi-hard sampling constructed {len(triplets)} triplets in total (limit {max_triplets})")
    return triplets


def similarity_matching(input_gz: str,
                        use_semi_hard: bool = False,
                        margin: float = 0.2,
                        max_triplets: int = 10000
                       ) -> List[Tuple[Dict[int,int],Dict[int,int],Dict[int,int]]]:
    # 1) Read bbvs
    pat = re.compile(r":(\d+):(\d+)")
    bbvs: List[Dict[int,int]] = []
    with gzip.open(input_gz, 'rt', encoding='utf-8', errors='replace') as inp:
        for line in inp:
            line = line.strip()
            if not line or not line.startswith('T'):
                continue
            d = {int(k): int(v) for k, v in pat.findall(line)}
            bbvs.append(d)
    n = len(bbvs)
    if n < 2:
        print("Too few samples, cannot construct triplets", file=sys.stderr)
        sys.exit(1)

    # 2) Random projection
    max_id = max(k for d in bbvs for k in d.keys())
    proj = np.random.randn(max_id+1, 100)
    vectors = np.vstack([
        np.zeros(max_id+1, dtype=float)[:] .__setitem__(slice(None), 0) or
        np.array([d.get(i,0) for i in range(max_id+1)]).dot(proj)
        for d in bbvs
    ])

    # 3) Cosine similarity
    sim = cosine_similarity(vectors)

    # 4) Statistical threshold
    sims = sim[~np.eye(n, dtype=bool)]
    mu, sigma = sims.mean(), sims.std()
    pos_thresh = mu + 0.5 * sigma
    neg_thresh = mu - 1.0 * sigma
    print(f"μ={mu:.4f}, σ={sigma:.4f}, pos>{pos_thresh:.4f}, neg<{neg_thresh:.4f}")

    # 5) Construct triplets
    if use_semi_hard:
        dataset = build_semi_hard_triplets(sim, bbvs, pos_thresh, margin, max_triplets)
    else:
        dataset = build_triplets(sim, bbvs, pos_thresh, neg_thresh, max_triplets)

    # # 6) Clustering + visualization
    # dist = 1 - sim
    # n_clusters = min(30, n // 10 + 1)
    # clust = AgglomerativeClustering(
    #     n_clusters=n_clusters,
    #     metric='precomputed',
    #     linkage='average'
    # )
    # labels = clust.fit_predict(dist)
    # clusters_id: List[List[int]] = [[] for _ in range(n_clusters)]
    # for idx, lbl in enumerate(labels):
    #     clusters_id[lbl].append(idx)
    # plot_tsne_clusters(vectors, labels, clusters_id, input_gz + "_tsne.pdf")

    return dataset


def transform(
    dataset: List[Tuple[Dict[int,int],Dict[int,int],Dict[int,int]]],
    output: str,
    id_map: Dict[int, np.ndarray],
) -> None:
    new_data = []
    print(f"[Transform] id_map key range: {min(id_map)}–{max(id_map)}")
    for a, p, n in dataset:
        def to_emb_w(d: Dict[int,int]):
            items = sorted(d.items(), key=lambda kv: kv[1])
            keys, ws = zip(*items)
            embs = [id_map[k-1] for k in keys]
            return embs, list(ws)
        ae, aw = to_emb_w(a)
        pe, pw = to_emb_w(p)
        ne, nw = to_emb_w(n)
        new_data.append((ae, aw, pe, pw, ne, nw))

    with open(output, 'wb') as f:
        pickle.dump(new_data, f)
    print(f"[Transform] Writing {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',  '-i', required=True, help='Input .gz file')
    p.add_argument('--map',    '-m', help='bb_id→emb pickle file (default: <input>.vectors.pkl)')
    p.add_argument('--output', '-o', help='Output triplets pickle file (default: <input>.emb.pkl)')
    p.add_argument('--semi-hard', action='store_true',
                   help='Enable semi-hard negative sample mining')
    p.add_argument('--margin', type=float, default=0.25,
                   help='Margin for semi-hard mining')
    p.add_argument('--max-triplets', type=int, default=10000,
                   help='Maximum number of triplets to sample per file (default 10000)')
    args = p.parse_args()

    base, _ = os.path.splitext(args.input)
    print(f"Processing: {base}")
    if not args.map:
        args.map = base + ".vectors.pkl"
    if not args.output:
        args.output = base + ".emb.pkl"

    try:
        id_map = load_map(args.map)
    except Exception as e:
        sys.stderr.write(f"Error: Unable to load map file {args.map}: {e}\n")
        traceback.print_exc()
        sys.exit(1)

    try:
        triplets = similarity_matching(
            args.input,
            use_semi_hard=args.semi_hard,
            margin=args.margin,
            max_triplets=args.max_triplets
        )
        print(f"[Main] Constructed {len(triplets)} triplets (limit {args.max_triplets})")
        transform(triplets, args.output, id_map)
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()