import numpy as np

from src.data.cleaning import clean_boxes


def test_clip_keeps_visible_part():
    b, e = clean_boxes(np.array([[1000, 10, 1060, 50]]), 1024, 1024, 2)
    assert b.tolist() == [[1000, 10, 1024, 50]]
    assert [x[1] for x in e] == ["clipped"]


def test_drop_outside_and_tiny():
    boxes = np.array([[1030, 10, 1060, 50], [10, 10, 11, 40], [10, 10, 50, 50]])
    b, e = clean_boxes(boxes, 1024, 1024, 2)
    assert len(b) == 1 and {x[1] for x in e} == {"dropped_outside", "dropped_tiny"}


def test_box_that_becomes_a_sliver_is_dropped():
    b, e = clean_boxes(np.array([[1023, 10, 1060, 50]]), 1024, 1024, 2)   # queda 1 px visible
    assert len(b) == 0 and e[0][1] == "dropped_tiny"


def test_inside_boxes_unchanged_and_no_events():
    boxes = np.array([[0, 0, 1024, 1024], [10, 20, 30, 40]], dtype=float)
    b, e = clean_boxes(boxes, 1024, 1024, 2)
    assert np.array_equal(b, boxes.astype(np.float32)) and e == []


def test_idempotent():
    boxes = np.array([[-5, 0, 20, 20], [1000, 1000, 1100, 1100]])
    b1, _ = clean_boxes(boxes, 1024, 1024, 2)
    b2, e2 = clean_boxes(b1, 1024, 1024, 2)
    assert np.array_equal(b1, b2) and e2 == []


def test_non_finite_and_empty():
    b, e = clean_boxes(np.array([[np.nan, 0, 10, 10]]), 1024, 1024, 2)
    assert len(b) == 0 and e[0][1] == "dropped_non_finite"
    b, e = clean_boxes(np.zeros((0, 4)), 1024, 1024, 2)
    assert b.shape == (0, 4) and e == []