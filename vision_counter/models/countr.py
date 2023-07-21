import math
from typing import List, Union, Optional

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from vision_counter.utils import maybe_download

WEIGHTS = {
    'CounTR': 'https://github.com/tamnguyenvan/vision-counter-assets/releases/download/v0.1.0/FSC147.onnx'
}


class CounTR:
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the CounTR object.

        Parameters
        ----------
        model_path : str, optional
            Path to the ONNX model file. If not provided, the default model will be used.
        """
        if model_path is None:
            model_path = WEIGHTS['CounTR']
        self.model_path = maybe_download(model_path)
        self.session = ort.InferenceSession(
            self.model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

    def count(
        self,
        image: Union[str, np.ndarray, Image.Image],
        bboxes: List[int]
    ) -> int:
        """Estimate the count of objects within the specified bounding boxes in the input image.

        Parameters
        ----------
        image : Union[str, np.ndarray, Image.Image]
            The input image in one of the following formats:
            - str: The path to the image file.
            - np.ndarray: The image as a NumPy array.
            - Image.Image: The image as a PIL Image object.
        bboxes : List[int]
            List of bounding boxes represented as a list of integers [x1, y1, x2, y2].

        Returns
        -------
        int
            The estimated count of objects within the specified bounding boxes.
        """
        samples, boxes, pos = self._preprocess(image, bboxes)
        samples = np.expand_dims(samples, axis=0)
        boxes = np.expand_dims(boxes, axis=0)
        num_shots = np.array([len(bboxes)], dtype=np.int64)

        _, _, h, w = samples.shape

        # Inference
        size = 384
        density_map = np.zeros([h, w])
        start = 0
        prev = -1
        while start + size - 1 < w:
            output, = self.session.run(
                None,
                {
                    'samples': samples[:, :, :, start:start + size],
                    'boxes': boxes,
                    'num_shots': num_shots
                }
            )[0]

            d1 = np.pad(
                output[:, 0:prev - start + 1],
                [[0, 0], [start, w - prev - 1]]
            )
            d2 = np.pad(
                output[:, prev - start + 1:384],
                [[0, 0], [prev + 1, w - start - size]]
            )

            density_map_l = np.pad(
                density_map[:, 0:start],
                [[0, 0], [0, w - start]]
            )

            density_map_m = np.pad(
                density_map[:, start:prev + 1],
                [[0, 0], [start, w - prev - 1]]
            )
            density_map_r = np.pad(
                density_map[:, prev + 1:w],
                [[0, 0], [prev + 1, 0]]
            )

            density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2

            prev = start + size - 1
            start = start + 128
            if start + size - 1 >= w:
                if start == w - size + 128:
                    break
                else:
                    start = w - size

        pred_cnt = np.sum(density_map / 60)
        e_cnt = 0
        for rect in pos:
            e_cnt += np.sum(density_map[rect[0]:rect[2] + 1, rect[1]:rect[3] + 1] / 60).item()
        e_cnt = e_cnt / 3
        if e_cnt > 1.8:
            pred_cnt /= e_cnt
        return math.ceil(pred_cnt)

    def _preprocess(
        self,
        image: Union[str, np.ndarray, Image.Image],
        bboxes: List[List[int]]
    ) -> np.ndarray:
        """Preprocess the input image and bounding boxes.

        Parameters
        ----------
        image : np.ndarray
            The input image as a NumPy array.

        Returns
        -------
        np.ndarray
            The preprocessed image as a NumPy array.
        """
        if isinstance(image, str):
            image = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            image = np.array(image)

        image_h, image_w = image.shape[:2]

        # Resize the image size so that the height is 384
        new_h = 384
        new_w = 16 * int((image_w / image_h * 384) / 16)
        scale_factor_h = float(new_h) / image_h
        scale_factor_w = float(new_w) / image_w
        image = cv2.resize(image, (new_w, new_h))
        image = image.astype(np.float32) / 255.0

        boxes = list()
        rects = list()
        for bbox in bboxes:
            x1 = int(bbox[0][0] * scale_factor_w)
            y1 = int(bbox[0][1] * scale_factor_h)
            x2 = int(bbox[1][0] * scale_factor_w)
            y2 = int(bbox[1][1] * scale_factor_h)
            rects.append([y1, x1, y2, x2])
            bbox = image[y1:y2 + 1, x1:x2 + 1, :]
            bbox = cv2.resize(bbox, (64, 64))
            bbox = np.transpose(bbox, (2, 0, 1))
            boxes.append(bbox)

        boxes = np.array(boxes).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        return image, boxes, rects