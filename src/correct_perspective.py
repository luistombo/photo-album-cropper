from __future__ import annotations

import cv2
import numpy as np


def order_points(points: np.ndarray) -> np.ndarray:
    points = points.astype("float32")
    rect = np.zeros((4, 2), dtype="float32")

    sums = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]

    diffs = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def add_margin_to_quad(points: np.ndarray, image_shape: tuple[int, int, int], margin_percent: float) -> np.ndarray:
    if margin_percent <= 0:
        return points.astype("float32")

    height, width = image_shape[:2]
    center = points.mean(axis=0)
    expanded = points.astype("float32").copy()
    for index, point in enumerate(expanded):
        vector = point - center
        expanded[index] = point + vector * (margin_percent / 100.0)

    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    return expanded


def warp_perspective(
    image: np.ndarray,
    points: np.ndarray,
    margin_percent: float = 0.0,
    points_are_ordered: bool = False,
) -> np.ndarray:
    points = add_margin_to_quad(points, image.shape, margin_percent)
    rect = points.astype("float32") if points_are_ordered else order_points(points)
    top_left, top_right, bottom_right, bottom_left = rect

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    target_width = int(round(max(width_top, width_bottom)))

    height_right = np.linalg.norm(bottom_right - top_right)
    height_left = np.linalg.norm(bottom_left - top_left)
    target_height = int(round(max(height_right, height_left)))

    if target_width < 20 or target_height < 20:
        raise ValueError("El recorte detectado es demasiado pequeño.")

    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (target_width, target_height), flags=cv2.INTER_CUBIC)


def rotate_to_landscape_if_close(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if height > width * 1.15:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def rotate_by_degrees(image: np.ndarray, degrees_clockwise: int) -> np.ndarray:
    degrees_clockwise = degrees_clockwise % 360
    if degrees_clockwise == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees_clockwise == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees_clockwise == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image
