#!/usr/bin/env python3
"""Train and upload models to R2.

This script trains machine learning models for the Mahler trading system
and uploads them to Cloudflare R2 for use by the Workers runtime.

Usage:
    python scripts/train_models.py --all
    python scripts/train_models.py --regime
    python scripts/train_models.py --weights
    python scripts/train_models.py --exit
    python scripts/train_models.py --all --dry-run

Environment Variables Required:
    ALPACA_API_KEY - Alpaca API key for fetching historical bars
    ALPACA_SECRET_KEY - Alpaca secret key

    For D1 access (optional, for weight/exit optimization):
    CLOUDFLARE_API_TOKEN - Cloudflare API token with D1 read access
    CLOUDFLARE_ACCOUNT_ID - Cloudflare account ID
    D1_DATABASE_ID - D1 database ID

    For R2 upload (not needed with --dry-run):
    R2_ACCESS_KEY_ID - R2 access key ID
    R2_SECRET_ACCESS_KEY - R2 secret access key
    R2_ENDPOINT_URL - R2 endpoint URL (e.g., https://ACCOUNT_ID.r2.cloudflarestorage.com)
    R2_BUCKET_NAME - R2 bucket name (default: mahler-models)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from training.data_fetcher import fetch_spy_bars, fetch_trades_from_d1
from training.exit_trainer import train_exit_model
from training.regime_trainer import train_regime_model
from training.weight_trainer import train_weight_models


def upload_to_r2(key: str, data: dict) -> None:
    """Upload model data to R2.

    Args:
        key: Object key in R2
        data: Dictionary to upload as JSON
    """
    import boto3

    endpoint_url = os.environ["R2_ENDPOINT_URL"]
    access_key_id = os.environ["R2_ACCESS_KEY_ID"]
    secret_access_key = os.environ["R2_SECRET_ACCESS_KEY"]
    bucket_name = os.environ.get("R2_BUCKET_NAME", "mahler-models")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )

    body = json.dumps(data, indent=2).encode("utf-8")

    s3.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=body,
        ContentType="application/json",
    )

    print(f"Uploaded {key} to R2 bucket {bucket_name}")


async def main() -> int:
    """Main training entrypoint."""
    parser = argparse.ArgumentParser(
        description="Train models for Mahler trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all models",
    )
    parser.add_argument(
        "--regime",
        action="store_true",
        help="Train regime detection model",
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        help="Train weight optimization models",
    )
    parser.add_argument(
        "--exit",
        action="store_true",
        help="Train exit optimization model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train but don't upload to R2",
    )
    parser.add_argument(
        "--save-local",
        type=str,
        help="Save models to local directory (for testing)",
    )

    args = parser.parse_args()

    if not any([args.all, args.regime, args.weights, args.exit]):
        parser.error("Must specify at least one model to train (--all, --regime, --weights, --exit)")

    train_regime = args.all or args.regime
    train_weights = args.all or args.weights
    train_exit = args.all or args.exit

    # Check required environment variables
    if train_regime and (
        not os.environ.get("ALPACA_API_KEY") or not os.environ.get("ALPACA_SECRET_KEY")
    ):
        parser.error("ALPACA_API_KEY and ALPACA_SECRET_KEY are required for regime training")

    if not args.dry_run and not args.save_local:
        required_r2_vars = ["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL"]
        missing = [v for v in required_r2_vars if not os.environ.get(v)]
        if missing:
            parser.error(f"Missing R2 environment variables: {', '.join(missing)}")

    print("=" * 60)
    print("Mahler Model Training")
    print("=" * 60)

    # Fetch training data
    bars = []
    trades = []

    if train_regime:
        print("\nFetching SPY historical bars (3 years)...")
        bars = await fetch_spy_bars(days=750)

    if train_weights or train_exit:
        print("\nFetching trade history from D1...")
        trades = await fetch_trades_from_d1()

    # Train models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if train_regime:
        print("\n" + "-" * 40)
        print("Training regime detection model...")
        print("-" * 40)

        if len(bars) < 60:
            print(f"ERROR: Insufficient bars for training: {len(bars)} < 60")
            return 1

        regime_params = train_regime_model(bars)

        if not args.dry_run:
            if args.save_local:
                save_path = Path(args.save_local) / "regime_model.json"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w") as f:
                    json.dump(regime_params, f, indent=2)
                print(f"Saved to {save_path}")
            else:
                upload_to_r2("models/regime/latest.json", regime_params)
                upload_to_r2(f"models/regime/{timestamp}.json", regime_params)

    if train_weights:
        print("\n" + "-" * 40)
        print("Training weight optimization models...")
        print("-" * 40)

        weight_params = train_weight_models(trades)

        if weight_params:
            if not args.dry_run:
                if args.save_local:
                    save_path = Path(args.save_local) / "weight_model.json"
                    with open(save_path, "w") as f:
                        json.dump(weight_params, f, indent=2)
                    print(f"Saved to {save_path}")
                else:
                    upload_to_r2("models/weights/latest.json", weight_params)
                    upload_to_r2(f"models/weights/{timestamp}.json", weight_params)
        else:
            print("Skipping weight model upload (insufficient data)")

    if train_exit:
        print("\n" + "-" * 40)
        print("Training exit optimization model...")
        print("-" * 40)

        exit_params = train_exit_model(trades)

        if exit_params:
            if not args.dry_run:
                if args.save_local:
                    save_path = Path(args.save_local) / "exit_model.json"
                    with open(save_path, "w") as f:
                        json.dump(exit_params, f, indent=2)
                    print(f"Saved to {save_path}")
                else:
                    upload_to_r2("models/exit/latest.json", exit_params)
                    upload_to_r2(f"models/exit/{timestamp}.json", exit_params)
        else:
            print("Skipping exit model upload (insufficient data)")

    print("\n" + "=" * 60)
    print("Training complete!")
    if args.dry_run:
        print("(Dry run - no models were uploaded)")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
