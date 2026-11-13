import json

import pytest

from energy.data import io


@pytest.mark.parametrize(
    "writer,payload",
    [
        (io.write_bytes_atomic, b"\x00\x01binary"),
        (io.write_json_atomic, {"a": [1, 2, 3], "b": "ñ"}),
    ],
)
def test_write_creates_parent_dirs(tmp_path, writer, payload):
    target = tmp_path / "x" / "y" / "z" / "file.dat"
    writer(target, payload)
    assert target.exists()


@pytest.mark.parametrize(
    "writer,payload",
    [
        (io.write_bytes_atomic, b"abc"),
        (io.write_json_atomic, {"k": 1}),
    ],
)
def test_write_leaves_no_tmp_file(tmp_path, writer, payload):
    target = tmp_path / "file.json"
    writer(target, payload)
    assert [p.name for p in tmp_path.iterdir()] == ["file.json"]


def test_write_bytes_round_trips(tmp_path):
    target = tmp_path / "f.bin"
    data = bytes(range(256))
    io.write_bytes_atomic(target, data)
    assert target.read_bytes() == data


def test_write_json_round_trips(tmp_path):
    target = tmp_path / "f.json"
    obj = {"items": [{"x": 1.5}, {"y": None}], "s": "texto"}
    io.write_json_atomic(target, obj)
    assert json.loads(target.read_text(encoding="utf-8")) == obj


def test_write_bytes_overwrites_existing_file(tmp_path):
    target = tmp_path / "f.bin"
    io.write_bytes_atomic(target, b"old content that is longer")
    io.write_bytes_atomic(target, b"new")
    assert target.read_bytes() == b"new"


def test_write_json_overwrites_existing_file(tmp_path):
    target = tmp_path / "f.json"
    io.write_json_atomic(target, {"v": 1})
    io.write_json_atomic(target, {"v": 2})
    assert json.loads(target.read_text()) == {"v": 2}
