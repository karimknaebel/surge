from functools import lru_cache

import cv2
import numpy as np
from huggingface_hub import hf_hub_download

SKYSEG_MODEL = "JianyuanWang/skyseg"
SKYSEG_FILENAME = "skyseg.onnx"


@lru_cache(maxsize=1)
def skyseg_session():
    import onnxruntime

    return onnxruntime.InferenceSession(hf_hub_download(SKYSEG_MODEL, SKYSEG_FILENAME))


def skyseg_mask(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (320, 320))
    x = resized.astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )
    x = x.transpose(2, 0, 1)[None].astype(np.float32)

    session = skyseg_session()
    result = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: x},
    )[0].squeeze()
    result = (result - result.min()) / (result.max() - result.min()) * 255
    result = cv2.resize(result.astype(np.uint8), (image.shape[1], image.shape[0]))
    return result < 32


def apply_background_filters(
    result: dict[str, np.ndarray],
    filter_sky: bool,
    filter_black_background: bool,
    filter_white_background: bool,
) -> dict[str, np.ndarray]:
    if not (filter_sky or filter_black_background or filter_white_background):
        return result

    mask = result["mask"].copy()
    image = result["image"]
    if filter_sky:
        mask &= skyseg_mask(image)
    if filter_black_background:
        mask &= image.sum(axis=-1) >= 16
    if filter_white_background:
        mask &= ~((image[..., 0] > 240) & (image[..., 1] > 240) & (image[..., 2] > 240))
    return {**result, "mask": mask}
