#!/usr/bin/env python3
"""
generate_groundtruth.py

For each dataset CSV in `--data_dir`, loads its features and the corresponding
init centers CSV from `--init_dir`, fits sklearn KMeans with those inits,
and writes out:

  1) <base>_gt_labels.csv    ← one-column CSV of predicted labels
  2) <base>_gt_centers.csv   ← k×d CSV of centroid coordinates

Usage:
  python generate_groundtruth.py \
    --data_dir kmeans_datasets_csv \
    --init_dir kmeans_inits \
    --out_dir kmeans_groundtruth \
    --init_seed 1234
"""
import argparse
import os
import numpy as np
from glob import glob
from sklearn.cluster import KMeans

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    pattern = os.path.join(args.data_dir, "*.csv")
    for data_path in glob(pattern):
        base = os.path.basename(data_path).rsplit(".", 1)[0]
        init_path = os.path.join(
            args.init_dir,
            f"{base}_init_seed{args.init_seed}.csv"
        )

        if not os.path.exists(init_path):
            print(f"[WARN] init file not found for {base}, skipping")
            continue

        # --- load dataset (features + true labels) ---
        arr = np.loadtxt(data_path, delimiter=",", skiprows=1)
        X = arr[:, :-1]  # drop last column (true label)

        # --- load init centers ---
        centers_init = np.loadtxt(init_path, delimiter=",", skiprows=1)
        k, d = centers_init.shape

        # --- run sklearn KMeans with fixed init ---
        km = KMeans(
            n_clusters=k,
            init=centers_init,
            n_init=1,
            random_state=args.init_seed
        ).fit(X)

        # --- save predicted labels ---
        labels = km.labels_
        labels_out = os.path.join(args.out_dir, f"{base}_gt_labels.csv")
        np.savetxt(
            labels_out,
            labels,
            fmt="%d",
            delimiter=",",
            header="label",
            comments=""
        )

        # --- save learned centers ---
        centers_out = os.path.join(args.out_dir, f"{base}_gt_centers.csv")
        header = ",".join(f"f{i}" for i in range(d))
        np.savetxt(
            centers_out,
            km.cluster_centers_,
            delimiter=",",
            header=header,
            comments=""
        )

        print(f"[OK] {base} → labels: {labels_out}, centers: {centers_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", required=True,
        help="Directory of dataset CSVs (each with d features + true label col)."
    )
    parser.add_argument(
        "--init_dir", required=True,
        help="Directory of CSVs containing initial centroids."
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Where to write the groundtruth label/center CSVs."
    )
    parser.add_argument(
        "--init_seed", type=int, default=1234,
        help="Seed used when generating the init centroids (to match filenames)."
    )
    args = parser.parse_args()
    main(args)

