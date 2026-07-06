from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.correct_perspective import order_points


GEMINI_ENDPOINT_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def detect_photo_quad_with_gemini(
    image_path: Path,
    image_shape: tuple[int, int, int],
    config: dict[str, Any],
) -> tuple[np.ndarray | None, str, int]:
    gemini_config = config.get("gemini", {})
    api_key = _get_api_key()
    if not api_key:
        logging.warning("Gemini esta activado, pero GEMINI_API_KEY no esta disponible.")
        return None, "gemini_no_key", 0

    model = str(gemini_config.get("model", "gemini-3.5-flash"))
    min_confidence = float(gemini_config.get("min_confidence", 0.55))
    max_retries = int(gemini_config.get("max_retries", 2))
    retry_delay_seconds = float(gemini_config.get("retry_delay_seconds", 2.0))
    request_timeout_seconds = float(gemini_config.get("request_timeout_seconds", 25.0))
    prompt = _build_prompt(image_shape)
    payload = _build_payload(image_path, prompt)

    body = ""
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            GEMINI_ENDPOINT_TEMPLATE.format(model=model),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=request_timeout_seconds) as response:
                body = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            if error.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                delay = retry_delay_seconds * (attempt + 1)
                logging.info("Gemini HTTP %s, reintentando en %.1fs", error.code, delay)
                time.sleep(delay)
                continue
            logging.warning("Gemini HTTP %s: %s", error.code, details[:400])
            return None, "gemini_error", 0
        except (TimeoutError, urllib.error.URLError) as error:
            if attempt < max_retries:
                delay = retry_delay_seconds * (attempt + 1)
                logging.info("Gemini no respondio, reintentando en %.1fs: %s", delay, error)
                time.sleep(delay)
                continue
            logging.warning("Gemini no respondio: %s", error)
            return None, "gemini_error", 0

    text = _response_text(json.loads(body))
    if not text:
        return None, "gemini_empty", 0

    parsed = _extract_json_object(text)
    if not parsed:
        logging.warning("Gemini devolvio una respuesta no JSON: %s", text[:300])
        return None, "gemini_bad_json", 0

    confidence = _parse_confidence(parsed)
    if confidence < min_confidence:
        logging.info("Gemini descarto deteccion por baja confianza: %.2f", confidence)
        return None, "gemini_low_confidence", 0

    points = parsed.get("points_1000") or parsed.get("points")
    quad = _points_to_quad(points, image_shape)
    if quad is None:
        logging.warning("Gemini no devolvio cuatro puntos validos: %s", parsed)
        return None, "gemini_bad_points", 0

    if not _is_valid_gemini_quad(quad, image_shape):
        logging.warning("Gemini devolvio un cuadrilatero no confiable: %s", quad.round(1).tolist())
        return None, "gemini_unreasonable", 0

    return quad, f"gemini:{confidence:.2f} semantic", 0


def detect_upright_rotation_with_gemini(
    image_bgr: np.ndarray,
    config: dict[str, Any],
) -> tuple[int, str]:
    gemini_config = config.get("gemini", {})
    api_key = _get_api_key()
    if not api_key:
        return 0, "orientation_no_key"

    model = str(gemini_config.get("model", "gemini-3.5-flash"))
    min_confidence = float(gemini_config.get("orientation_min_confidence", 0.65))
    max_retries = int(gemini_config.get("orientation_max_retries", 1))
    retry_delay_seconds = float(gemini_config.get("retry_delay_seconds", 2.0))
    request_timeout_seconds = float(gemini_config.get("request_timeout_seconds", 25.0))
    prompt = (
        "This is a cropped old printed photograph. Decide how many degrees clockwise it must be rotated so the scene "
        "is upright. Use people standing/sitting naturally, heads above feet, horizon/sea/sky above ground, buildings "
        "and trees vertical, cars/furniture/text orientation. Return only strict JSON with this schema: "
        '{"rotation_degrees_clockwise":0,"confidence":0.0,"reason":"short"}. '
        "rotation_degrees_clockwise must be exactly one of 0, 90, 180, 270."
    )
    payload = _build_payload_from_bytes(_image_array_bytes_for_gemini(image_bgr), "image/jpeg", prompt)

    for attempt in range(max_retries + 1):
        request = _gemini_request(model, api_key, payload)
        try:
            with urllib.request.urlopen(request, timeout=request_timeout_seconds) as response:
                body = response.read().decode("utf-8")
            parsed = _extract_json_object(_response_text(json.loads(body)))
            if not parsed:
                return 0, "orientation_bad_json"
            confidence = _parse_confidence(parsed)
            if confidence < min_confidence:
                return 0, f"orientation_low_confidence:{confidence:.2f}"
            rotation = _parse_rotation(parsed)
            return rotation, f"orientation:{confidence:.2f}"
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as error:
            if attempt < max_retries:
                delay = retry_delay_seconds * (attempt + 1)
                logging.info("Gemini orientacion no respondio, reintentando en %.1fs: %s", delay, error)
                time.sleep(delay)
                continue
            logging.info("Gemini orientacion no disponible: %s", error)
            return 0, "orientation_error"

    return 0, "orientation_error"


def _build_prompt(image_shape: tuple[int, int, int]) -> str:
    height, width = image_shape[:2]
    return (
        "Detect the four visible corners of the ONE printed photograph that should be cropped from this album-page "
        "snapshot. Exclude album page borders, plastic sleeve edges, table/background, blank separator bands, and "
        "adjacent partial photos. If there are multiple photos visible, choose the main complete printed photo in the "
        "center/largest usable area, not a partial photo above or below. Return only strict JSON, no markdown. "
        "Coordinates must be normalized integers from 0 to 1000 relative to the full input image. CRITICAL: order the "
        "four points by the final upright photograph content, not by screen position. The first point must be the top-left "
        "corner of the printed photo when the scene is viewed upright, then top-right, bottom-right, bottom-left. This means "
        "if the phone photo is rotated or upside down, the first point may visually appear on the right, bottom, or elsewhere "
        "in the input image. Use human heads above feet, sky above ground, water/horizon level, buildings/trees vertical, "
        "and readable text to decide the upright orientation. Return only strict JSON, no markdown. Use this exact schema: "
        '{"points_1000":[{"x":0,"y":0},{"x":1000,"y":0},{"x":1000,"y":1000},{"x":0,"y":1000}],'
        '"confidence":0.0,"reason":"short"}'
        f" The source image size is {width}x{height} pixels."
    )


def _build_payload(image_path: Path, prompt: str) -> dict[str, Any]:
    mime_type, image_bytes = _image_bytes_for_gemini(image_path)
    return _build_payload_from_bytes(image_bytes, mime_type, prompt)


def _build_payload_from_bytes(image_bytes: bytes, mime_type: str, prompt: str) -> dict[str, Any]:
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    return {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": image_b64}},
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    }


def _gemini_request(model: str, api_key: str, payload: dict[str, Any]) -> urllib.request.Request:
    return urllib.request.Request(
        GEMINI_ENDPOINT_TEMPLATE.format(model=model),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )


def _image_bytes_for_gemini(image_path: Path) -> tuple[str, bytes]:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=86, optimize=True)
            return "image/jpeg", buffer.getvalue()
    except Exception:
        return mime_type, image_path.read_bytes()


def _image_array_bytes_for_gemini(image_bgr: np.ndarray) -> bytes:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=86, optimize=True)
    return buffer.getvalue()


def _response_text(response: dict[str, Any]) -> str:
    parts = response.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(str(part.get("text", "")) for part in parts).strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _points_to_quad(points: Any, image_shape: tuple[int, int, int]) -> np.ndarray | None:
    if not isinstance(points, list) or len(points) != 4:
        return None

    height, width = image_shape[:2]
    converted: list[list[float]] = []
    for point in points:
        if not isinstance(point, dict) or "x" not in point or "y" not in point:
            return None
        x = float(point["x"])
        y = float(point["y"])
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            return None
        converted.append([x * (width - 1) / 1000.0, y * (height - 1) / 1000.0])
    return np.array(converted, dtype=np.float32)


def _parse_rotation(parsed: dict[str, Any]) -> int:
    raw_rotation = parsed.get("rotation_degrees_clockwise", parsed.get("rotation", 0))
    try:
        rotation = int(raw_rotation)
    except (TypeError, ValueError):
        return 0
    if rotation not in {0, 90, 180, 270}:
        return 0
    return rotation


def _parse_confidence(parsed: dict[str, Any]) -> float:
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
    if 1.0 < confidence <= 100.0:
        confidence /= 100.0
    return clamp_float(confidence, 0.0, 1.0)


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _is_valid_gemini_quad(quad: np.ndarray, image_shape: tuple[int, int, int]) -> bool:
    height, width = image_shape[:2]
    ordered = order_points(quad)
    area = cv2.contourArea(ordered.astype(np.float32))
    image_area = height * width
    if area < image_area * 0.08 or area > image_area * 1.02:
        return False

    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])
    min_side = min(top_width, bottom_width, left_height, right_height)
    max_side = max(top_width, bottom_width, left_height, right_height)
    if min_side < 40 or max_side / max(min_side, 1.0) > 8:
        return False

    aspect = max(top_width, bottom_width) / max(max(left_height, right_height), 1.0)
    return 0.2 <= aspect <= 5.5


def _get_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key:
        return api_key

    if os.name != "nt":
        return ""

    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, "GEMINI_API_KEY")
            return str(value).strip()
    except OSError:
        return ""
