import json

import pytest

from lab.config import load_config
from lab.labels import box_to_pixel, build_prompt, objects_to_answer, yolo_to_box, yolo_to_pixel
from scripts.score import parse_output


def test_yolo_to_box_norm1000():
    assert yolo_to_box("pothole", [0.5, 0.5, 0.2, 0.4]) == {"label": "pothole", "bbox_2d": [400, 300, 600, 700]}


def test_yolo_to_box_clamps_and_pixel_scale():
    assert yolo_to_box("crack", [0.05, 0.5, 0.2, 0.1])["bbox_2d"][0] == 0
    assert yolo_to_box("crack", [0.5, 0.5, 0.5, 0.5], "pixel", 640, 360)["bbox_2d"] == [160, 90, 480, 270]


def test_tiny_box_stays_valid_for_scorer():
    ans = objects_to_answer([{"label": "crack", "yolo": [0.5, 0.5, 0.0002, 0.0002]}])
    assert parse_output(ans) is not None


def test_answer_sorted_compact_and_parseable():
    objs = [{"label": "manhole", "yolo": [0.8, 0.8, 0.1, 0.1]},
            {"label": "crack", "yolo": [0.2, 0.2, 0.1, 0.1]}]
    ans = objects_to_answer(objs)
    assert " " not in ans
    assert [b["label"] for b in json.loads(ans)] == ["crack", "manhole"]
    assert objects_to_answer([]) == "[]"
    assert len(parse_output(ans)) == 2


@pytest.mark.parametrize("coord", ["norm1000", "pixel"])
def test_round_trip_to_pixel(coord):
    yolo = [0.31, 0.62, 0.12, 0.08]
    box = yolo_to_box("pothole", yolo, coord, 640, 360)["bbox_2d"]
    for a, b in zip(box_to_pixel(box, coord, 640, 360), yolo_to_pixel(yolo, 640, 360)):
        assert abs(a - b) <= 0.7


def test_prompt_mentions_scale():
    assert "0-1000" in build_prompt("norm1000")
    assert "640x360" in build_prompt("pixel", 640, 360)
    with pytest.raises(ValueError):
        build_prompt("bad")


def test_config_overrides():
    cfg = load_config("configs/e2.yaml", ["lora.r=64", "train_fraction=0.1"])
    assert cfg["lora"]["r"] == 64 and cfg["lora"]["alpha"] == 128
    assert cfg["train_fraction"] == 0.1 and cfg["train_projector"] is True
    assert load_config("configs/e0.yaml")["lora"] is None
