import numpy as np
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture()
def synthetic_split(tmp_path):
    """6 imágenes PNG sintéticas de 96x96 (una sin espigas) con su partición y anotaciones."""
    rng = np.random.default_rng(0)
    images = tmp_path / "images"
    images.mkdir()
    names = [f"img{i}.png" for i in range(6)]
    boxes = {names[0]: "10 10 40 40;50 50 90 80", names[1]: "5 5 30 30", names[2]: "no_box",
             names[3]: "20 20 60 70;0 0 25 25;70 10 95 40", names[4]: "30 30 50 60", names[5]: "10 60 40 90"}
    for n in names:
        Image.fromarray(rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)).save(images / n)
    pd.DataFrame({"image_name": names, "group": [f"g{i % 2}" for i in range(6)],
                  "domain": [f"d{i % 3}" for i in range(6)]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"image_name": names, "BoxesString": [boxes[n] for n in names],
                  "n_boxes": [0] * 6}).to_csv(tmp_path / "annotations_clean.csv", index=False)
    return {"split_csv": tmp_path / "train.csv", "ann_csv": tmp_path / "annotations_clean.csv",
            "images_dir": images, "names": names, "boxes": boxes}