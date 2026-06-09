"""CLI entry point for the AI CTO Daily scheduler."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from ..models import ScheduleConfig
from ..storage.manager import StorageManager
from .scheduler import run_forever


ROOT = Path(__file__).resolve().parents[2]


def load_schedule_config() -> ScheduleConfig:
    load_dotenv(ROOT / ".env", override=False)
    storage = StorageManager(data_dir=str(ROOT / "data"))
    return storage.load_config().schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="AI CTO Daily local scheduler")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    os.chdir(ROOT)
    run_forever(load_schedule_config, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
