"""Tests for the label helpers in `energy.app.model`."""
from datetime import date

import pytest

from energy.app.model import period_label, status_line


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (date(2025, 10, 1), date(2025, 12, 31), "oct–dic 2025"),
        (date(2025, 10, 1), date(2025, 10, 31), "oct 2025"),
        (date(2024, 2, 1), date(2024, 2, 29), "feb 2024"),
        (date(2025, 10, 6), date(2025, 10, 10), "6 oct – 10 oct 2025"),
        (date(2025, 10, 6), date(2025, 12, 31), "6 oct – 31 dic 2025"),
    ],
)
def test_period_label(start, end, expected):
    assert period_label(start, end) == expected


def test_status_line():
    line = status_line("8h", date(2025, 10, 1), date(2025, 12, 31), "default")
    assert line == "Batería de 8 h · oct–dic 2025 · Operación: Mixto"


def test_status_line_other_mix():
    assert status_line("4h", date(2024, 10, 1), date(2024, 12, 31), "mostly_single").startswith(
        "Batería de 4 h · oct–dic 2024 · Operación: Principalmente un turno"
    )
