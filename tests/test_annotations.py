import numpy as np

from src.data.annotations import boxes_to_string, parse_boxes_string, validate_boxes


def test_parse_normal():
    b, e = parse_boxes_string("10 20 30 40;50 60 70 80")
    assert e == [] and b.shape == (2, 4) and b[1, 3] == 80


def test_parse_no_box_and_nan():
    assert parse_boxes_string("no_box")[0].shape == (0, 4)
    b, e = parse_boxes_string(float("nan"))
    assert b.shape == (0, 4) and e == ["missing_value"]


def test_parse_trailing_semicolon_and_commas():
    b, e = parse_boxes_string("1,2,3,4;")
    assert e == [] and b.shape == (1, 4)


def test_parse_errors():
    _, e1 = parse_boxes_string("1 2 3")
    _, e2 = parse_boxes_string("a b c d")
    assert e1 and e2


def test_roundtrip():
    b = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    b2, _ = parse_boxes_string(boxes_to_string(b))
    assert np.array_equal(b, b2)


def test_validate():
    boxes = np.array([[0, 0, 10, 10], [10, 10, 5, 20], [-3, 0, 10, 10], [0, 0, 1, 50]], dtype=np.float32)
    issues = {(i, k) for i, k, _ in validate_boxes(boxes, 100, 100, min_side=2)}
    assert (0, "non_positive_size") not in issues and (1, "non_positive_size") in issues
    assert (2, "out_of_bounds") in issues and (3, "tiny_box") in issues