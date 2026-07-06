from __future__ import annotations

import cv2
import numpy as np

from src.correct_perspective import order_points
from src.utils import resize_for_processing


def detect_photo_quad(
    image: np.ndarray,
    keep_white_border: bool = False,
    accept_already_cropped: bool = False,
) -> tuple[np.ndarray | None, str]:
    working, scale = resize_for_processing(image)
    quad, method = _detect_by_contours(working)
    if quad is None:
        quad, method = _detect_by_album_edges(working)
    if quad is None:
        quad, method = _detect_by_foreground_bbox(working)

    if quad is None:
        return None, "not_found"

    quad = quad.astype(np.float32) / scale
    if not _is_reasonable_quad(quad, image.shape):
        if accept_already_cropped and _is_nearly_full_frame(quad, image.shape):
            return _full_frame_quad(image.shape), "full_frame"
        return None, "unreasonable"

    if not keep_white_border:
        quad = _shrink_quad(quad, percent=0.8)

    return order_points(quad), method


def draw_debug_contour(image: np.ndarray, quad: np.ndarray | None, method: str) -> np.ndarray:
    debug = image.copy()
    if quad is not None:
        points = quad.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(debug, [points], True, (0, 255, 0), 5)
        for index, point in enumerate(quad.astype(int)):
            cv2.circle(debug, tuple(point), 12, (0, 0, 255), -1)
            cv2.putText(
                debug,
                str(index + 1),
                tuple(point + np.array([10, -10])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    cv2.putText(debug, f"method: {method}", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
    return debug


def _detect_by_contours(image: np.ndarray) -> tuple[np.ndarray | None, str]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]

    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred_lightness = cv2.GaussianBlur(lightness, (5, 5), 0)
    edges_gray = cv2.Canny(blurred_gray, 40, 130)
    edges_lightness = cv2.Canny(blurred_lightness, 35, 120)
    edges = cv2.bitwise_or(edges_gray, edges_lightness)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]
    image_area = height * width

    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.04 or area > image_area * 0.96:
            continue

        perimeter = cv2.arcLength(contour, True)
        for epsilon_ratio in (0.015, 0.02, 0.03, 0.04, 0.06):
            approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype(np.float32)
                if _is_reasonable_quad(quad, image.shape):
                    rectangularity = area / max(cv2.contourArea(approx), 1.0)
                    candidates.append((area * rectangularity, quad))
                break

    if not candidates:
        return None, "contours_failed"

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], "contours"


def _detect_by_album_edges(image: np.ndarray) -> tuple[np.ndarray | None, str]:
    height, width = image.shape[:2]
    row_score = _horizontal_edge_profile(image)
    candidates = _horizontal_line_candidates(image, row_score)

    top = _choose_top_album_edge(candidates, row_score, height, width)
    bottom = _choose_bottom_album_edge(candidates, row_score, height, width)
    if top is None:
        top = _edge_from_row_score(row_score, 0, int(height * 0.22), minimum_score=22.0)
    if bottom is None:
        bottom = _edge_from_row_score(row_score, int(height * 0.65), height, minimum_score=18.0)

    if top is None or bottom is None:
        return None, "album_edges_failed"

    if _line_y(bottom, width / 2) - _line_y(top, width / 2) < height * 0.42:
        return None, "album_edges_too_small"

    quad = np.array(
        [
            [0, _line_y(top, 0)],
            [width - 1, _line_y(top, width - 1)],
            [width - 1, _line_y(bottom, width - 1)],
            [0, _line_y(bottom, 0)],
        ],
        dtype=np.float32,
    )
    quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
    return quad, "album_edges"


def _horizontal_edge_profile(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    gradient = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3))
    profile = gradient.mean(axis=1)
    kernel_size = max(9, image.shape[0] // 100)
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(profile, kernel, mode="same")


def _horizontal_line_candidates(image: np.ndarray, row_score: np.ndarray) -> list[dict[str, float]]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    edges = cv2.Canny(cv2.GaussianBlur(equalized, (5, 5), 0), 35, 110)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=45,
        minLineLength=int(width * 0.15),
        maxLineGap=35,
    )
    if lines is None:
        return []

    segments: list[tuple[float, float, float, float, float, float]] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        angle = abs(float(np.degrees(np.arctan2(dy, dx))))
        if angle > 90:
            angle = 180 - angle
        if angle >= 8 or length < width * 0.15:
            continue
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        segments.append((float(x1), float(y1), float(x2), float(y2), length, (float(y1) + float(y2)) / 2))

    segments.sort(key=lambda segment: segment[5])
    clusters: list[list[tuple[float, float, float, float, float, float]]] = []
    for segment in segments:
        if clusters and abs(np.mean([item[5] for item in clusters[-1]]) - segment[5]) < 18:
            clusters[-1].append(segment)
        else:
            clusters.append([segment])

    candidates: list[dict[str, float]] = []
    for cluster in clusters:
        xs: list[float] = []
        ys: list[float] = []
        total_length = 0.0
        for x1, y1, x2, y2, length, _ in cluster:
            xs.extend([x1, x2])
            ys.extend([y1, y2])
            total_length += length

        xs_array = np.array(xs, dtype=np.float32)
        ys_array = np.array(ys, dtype=np.float32)
        if np.max(xs_array) - np.min(xs_array) < width * 0.12:
            continue

        slope, intercept = np.polyfit(xs_array, ys_array, 1)
        center_y = float(slope * (width - 1) / 2 + intercept)
        score_index = int(np.clip(round(center_y), 0, height - 1))
        candidates.append(
            {
                "slope": float(slope),
                "intercept": float(intercept),
                "center_y": center_y,
                "span": float(np.max(xs_array) - np.min(xs_array)),
                "total_length": total_length,
                "row_score": float(row_score[score_index]),
            }
        )
    return candidates


def _choose_top_album_edge(
    candidates: list[dict[str, float]], row_score: np.ndarray, height: int, width: int
) -> tuple[float, float] | None:
    top_limit = height * 0.22
    top_candidates = [candidate for candidate in candidates if 2 <= candidate["center_y"] <= top_limit]
    if not top_candidates:
        return None

    def score(candidate: dict[str, float]) -> float:
        edge_strength = candidate["row_score"] / 30.0
        coverage = candidate["span"] / width
        top_preference = 1.0 - candidate["center_y"] / top_limit
        return edge_strength + coverage + top_preference * 0.2

    selected = max(top_candidates, key=score)
    if selected["row_score"] < 18.0 and selected["span"] < width * 0.45:
        return None
    return selected["slope"], selected["intercept"]


def _choose_bottom_album_edge(
    candidates: list[dict[str, float]], row_score: np.ndarray, height: int, width: int
) -> tuple[float, float] | None:
    bottom_candidates = [candidate for candidate in candidates if candidate["center_y"] >= height * 0.60]
    if not bottom_candidates:
        return None

    def score(candidate: dict[str, float]) -> float:
        edge_strength = candidate["row_score"] / 30.0
        coverage = candidate["span"] / width
        length_bonus = min(candidate["total_length"] / (width * 3), 1.0)
        bottom_preference = (candidate["center_y"] - height * 0.60) / (height * 0.40)
        return edge_strength * 1.15 + coverage * 1.15 + length_bonus * 0.25 + bottom_preference * 1.2

    selected = max(bottom_candidates, key=score)
    if selected["center_y"] < height * 0.72:
        return None
    if selected["row_score"] < 12.0 and selected["span"] < width * 0.35:
        return None
    return selected["slope"], selected["intercept"]


def _edge_from_row_score(
    row_score: np.ndarray, start: int, end: int, minimum_score: float
) -> tuple[float, float] | None:
    start = max(0, start)
    end = min(len(row_score), end)
    if end <= start:
        return None

    local_index = int(np.argmax(row_score[start:end]))
    y = start + local_index
    if row_score[y] < minimum_score:
        return None
    return 0.0, float(y)


def _line_y(line: tuple[float, float], x: float) -> float:
    slope, intercept = line
    return slope * x + intercept


def _detect_by_foreground_bbox(image: np.ndarray) -> tuple[np.ndarray | None, str]:
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    saturation = hsv[:, :, 1]
    lightness = lab[:, :, 0]
    border_pixels = np.concatenate(
        [
            image[: max(2, height // 25), :, :].reshape(-1, 3),
            image[-max(2, height // 25) :, :, :].reshape(-1, 3),
            image[:, : max(2, width // 25), :].reshape(-1, 3),
            image[:, -max(2, width // 25) :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    border_color = np.median(border_pixels, axis=0)
    color_distance = np.linalg.norm(image.astype(np.float32) - border_color.astype(np.float32), axis=2)

    mask = ((color_distance > 22) | (saturation > 35) | (lightness < 225)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "foreground_failed"

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    image_area = height * width
    if area < image_area * 0.04:
        return None, "foreground_too_small"

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    if not _is_reasonable_quad(box, image.shape):
        x, y, w, h = cv2.boundingRect(contour)
        box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)

    return box, "foreground_bbox"


def _is_reasonable_quad(quad: np.ndarray, image_shape: tuple[int, int, int]) -> bool:
    height, width = image_shape[:2]
    image_area = height * width
    ordered = order_points(quad)
    area = cv2.contourArea(ordered.astype(np.float32))
    if area < image_area * 0.035 or area > image_area * 0.98:
        return False

    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])
    min_side = min(top_width, bottom_width, left_height, right_height)
    max_side = max(top_width, bottom_width, left_height, right_height)
    if min_side < 30 or max_side / max(min_side, 1.0) > 8:
        return False

    aspect = max(top_width, bottom_width) / max(max(left_height, right_height), 1.0)
    return 0.2 <= aspect <= 5.5


def _is_nearly_full_frame(quad: np.ndarray, image_shape: tuple[int, int, int]) -> bool:
    height, width = image_shape[:2]
    ordered = order_points(quad)
    area = cv2.contourArea(ordered.astype(np.float32))
    image_area = height * width
    if area < image_area * 0.92:
        return False

    tolerance = max(width, height) * 0.04
    expected = _full_frame_quad(image_shape)
    return bool(np.max(np.abs(ordered - expected)) <= tolerance)


def _full_frame_quad(image_shape: tuple[int, int, int]) -> np.ndarray:
    height, width = image_shape[:2]
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def _shrink_quad(quad: np.ndarray, percent: float) -> np.ndarray:
    center = quad.mean(axis=0)
    return center + (quad - center) * (1.0 - percent / 100.0)
