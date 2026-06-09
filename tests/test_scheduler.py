from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.models import ScheduleConfig
from src.services.scheduler import compute_next_run


def test_daily_schedule_uses_configured_timezone():
    config = ScheduleConfig(
        enabled=True,
        timezone="Asia/Shanghai",
        daily_times=["09:00"],
    )
    now = datetime(2026, 6, 9, 0, 30, tzinfo=timezone.utc)

    next_run = compute_next_run(config, now)

    assert next_run == datetime(2026, 6, 9, 1, 0, tzinfo=timezone.utc)


def test_daily_schedule_rolls_to_tomorrow():
    config = ScheduleConfig(
        enabled=True,
        timezone="Asia/Shanghai",
        daily_times=["09:00"],
    )
    now = datetime(2026, 6, 9, 2, 0, tzinfo=timezone.utc)

    next_run = compute_next_run(config, now)

    assert next_run == datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc)


def test_interval_schedule_uses_last_started_at():
    config = ScheduleConfig(enabled=True, interval_minutes=60)
    last_started_at = datetime(2026, 6, 9, 1, 0, tzinfo=timezone.utc)

    next_run = compute_next_run(
        config,
        datetime(2026, 6, 9, 1, 30, tzinfo=timezone.utc),
        last_started_at=last_started_at,
    )

    assert next_run == datetime(2026, 6, 9, 2, 0, tzinfo=timezone.utc)


def test_invalid_daily_time_is_rejected():
    with pytest.raises(ValidationError):
        ScheduleConfig(enabled=True, daily_times=["25:00"])
