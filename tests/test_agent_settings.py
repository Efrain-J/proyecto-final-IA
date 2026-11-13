"""Tests for `energy.agent.settings`: the user's battery and period (D16, option A).

Pure validation, no simulation. Run with ``uv run pytest tests/test_agent_settings.py``.
"""
from __future__ import annotations

import dataclasses
from datetime import date

import pytest

from energy import config
from energy.agent.settings import UserSettings


def test_defaults_equal_the_config_values():
    settings = UserSettings()
    assert settings.preset == config.DEFAULT_PRESET == "8h"
    assert settings.start == config.DEFAULT_START == date(2025, 10, 1)
    assert settings.end == config.DEFAULT_END == date(2025, 12, 31)


def test_settings_are_frozen():
    settings = UserSettings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.preset = "4h"  # type: ignore[misc]


@pytest.mark.parametrize("preset", ["4h", "8h", "12h"])
def test_valid_presets_are_accepted(preset):
    assert UserSettings(preset, date(2025, 1, 1), date(2025, 1, 31)).preset == preset


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        (("5h", date(2025, 10, 1), date(2025, 10, 31)), "5h"),
        (("8h", date(2025, 11, 1), date(2025, 10, 1)), ""),
        (("8h", date(2024, 12, 20), date(2025, 1, 5)), ""),
        (("8h", date(2023, 12, 1), date(2023, 12, 31)), ""),
        (("8h", date(2026, 1, 1), date(2026, 1, 31)), ""),
    ],
    ids=["bad-preset", "start-after-end", "crosses-year", "before-data", "after-data"],
)
def test_invalid_settings_raise_value_error(args, fragment):
    with pytest.raises(ValueError) as exc_info:
        UserSettings(*args)
    assert fragment in str(exc_info.value)


def test_data_boundaries_are_inclusive():
    UserSettings("8h", date(2024, 1, 1), date(2024, 12, 31))
    UserSettings("8h", date(2025, 1, 1), date(2025, 12, 31))


def test_single_day_period_is_valid():
    UserSettings("4h", date(2025, 5, 5), date(2025, 5, 5))
