#!/usr/bin/env python3
import argparse
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def load_labels(path, has_header):
    """
    Load a CSV of labels. If the CSV has multiple columns,
    we assume the last column is the label column.
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1 if has_header else 0)
    # if each row is a single value, data.ndim==1
    if data.ndim == 1:
        labels = data.astype(int)
    else:
        labels = data[:, -1].astype(int)
    return labels

def main():
    parser = argparse.ArgumentParser(
        description="Compare two cluster‑label CSVs via ARI and NMI"
    )
    parser.add_argument(
        "groundtruth_csv",
        help="Path to CSV with ground‑truth labels (one column or multiple; last column is label)."
    )
    parser.add_argument(
        "predicted_csv",
        help="Path to CSV with your algorithm’s predicted labels (same format)."
    )
    parser.add_argument(
        "--header", action="store_true",
        help="If set, skip the first row of each CSV (assumed header)."
    )
    args = parser.parse_args()

    y_true = load_labels(args.groundtruth_csv, args.header)
    y_pred = load_labels(args.predicted_csv, args.header)

    if y_true.shape[0] != y_pred.shape[0]:
        print(f"⚠️  Row count mismatch: ground‑truth has {y_true.shape[0]}, predicted has {y_pred.shape[0]}")
        exit(1)

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    print(f"Correctness metrics:")
    print(f"  • Adjusted Rand Index      : {ari:.4f}")
    print(f"  • Normalized Mutual Info    : {nmi:.4f}")

if __name__ == "__main__":
    main()
