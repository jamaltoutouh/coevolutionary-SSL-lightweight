#!/usr/bin/env python3
import argparse
import os

import openml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True, help="Root OpenML cache directory")
    ap.add_argument("--dataset_ids", type=int, nargs="+", default=None)
    ap.add_argument("--suite", default=None, help="e.g., OpenML-CC18")
    ap.add_argument("--max_datasets", type=int, default=9999)
    args = ap.parse_args()

    cache_dir = os.path.abspath(os.path.expanduser(args.cache_dir))
    os.makedirs(cache_dir, exist_ok=True)

    # Tell OpenML to use this cache (so you can later copy it to the cluster)
    openml.config.set_root_cache_directory(cache_dir)

    if args.dataset_ids is None:
        if not args.suite:
            raise SystemExit("Provide either --dataset_ids or --suite")
        suite = openml.study.get_suite(args.suite)
        dataset_ids = list(suite.data)[: args.max_datasets]
    else:
        dataset_ids = args.dataset_ids

    print(f"Using cache dir: {cache_dir}")
    print(f"Prefetching {len(dataset_ids)} datasets: {dataset_ids}")

    for did in dataset_ids:
        try:
            # Force downloading all relevant artifacts into cache.
            # (Newer OpenML versions can do lazy loading; explicit flags avoid missing files offline.) :contentReference[oaicite:1]{index=1}
            ds = openml.datasets.get_dataset(
                did,
                download_data=True,
                download_qualities=True,
                download_features_meta_data=True,
            )
            X, y, _, _ = ds.get_data(dataset_format="dataframe", target=ds.default_target_attribute)
            print(f"OK {did}: {ds.name}  shape={X.shape}")
        except Exception as e:
            print(f"FAIL {did}: {e}")

if __name__ == "__main__":
    main()
