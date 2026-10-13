import json

import pytest

from lab.reward import completion_text, gt_boxes, iou_reward

GT = gt_boxes([{"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}])
PERFECT = '[{"label":"pothole","bbox_2d":[400,400,600,600]}]'


def test_perfect():
    assert iou_reward(PERFECT, GT) == pytest.approx(1.0)


def test_fenced_json_still_valid():
    assert iou_reward("```json\n" + PERFECT + "\n```", GT) == pytest.approx(1.0)


def test_broken_json_is_zero():
    assert iou_reward('[{"label":"pothole","bbox_2d":[400,400', GT) == 0.0
    assert iou_reward("không có", GT) == 0.0


def test_wrong_class_only_gets_format_bonus():
    assert iou_reward('[{"label":"crack","bbox_2d":[400,400,600,600]}]', GT) == pytest.approx(0.2)


def test_extra_box_penalised_by_denominator():
    two = json.dumps([{"label": "pothole", "bbox_2d": [400, 400, 600, 600]},
                      {"label": "pothole", "bbox_2d": [0, 0, 100, 100]}])
    assert iou_reward(two, GT) == pytest.approx(0.6)


def test_huge_box_does_not_hack_reward():
    assert iou_reward('[{"label":"pothole","bbox_2d":[0,0,1000,1000]}]', GT) == pytest.approx(0.2)


def test_empty_cases():
    assert iou_reward("[]", []) == 1.0
    assert iou_reward("[]", GT) == pytest.approx(0.2)
    assert iou_reward(PERFECT, []) == pytest.approx(0.2)


def test_low_iou_not_matched():
    shifted = '[{"label":"pothole","bbox_2d":[500,500,700,700]}]'   # IoU = 1/7 < 0.5
    assert iou_reward(shifted, GT) == pytest.approx(0.2)


def test_conversational_completion():
    msg = [{"role": "assistant", "content": PERFECT}]
    assert completion_text(msg) == PERFECT
    assert iou_reward(msg, GT) == pytest.approx(1.0)
