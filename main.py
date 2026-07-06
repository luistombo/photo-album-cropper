from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:
    from tqdm import tqdm

    from src.color_correction import correct_color
    from src.correct_perspective import rotate_by_degrees, rotate_to_landscape_if_close, warp_perspective
    from src.detect_photo import detect_photo_quad, draw_debug_contour
    from src.gemini_detector import detect_photo_quad_with_gemini, detect_upright_rotation_with_gemini
    from src.utils import (
        copy_to_needs_review,
        ensure_output_dirs,
        iter_image_files,
        load_config,
        read_image,
        safe_output_name,
        save_image,
        setup_logging,
    )
except ModuleNotFoundError as error:
    missing = error.name or "una dependencia"
    print(f"Falta instalar {missing}. Ejecute: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detecta, endereza, recorta y corrige fotos impresas fotografiadas en albumes."
    )
    parser.add_argument("--input", default="input", type=Path, help="Carpeta con fotos originales.")
    parser.add_argument("--output", default="output", type=Path, help="Carpeta de salida.")
    parser.add_argument("--config", default=Path("config.json"), type=Path, help="Archivo JSON de configuracion.")
    return parser.parse_args()


def process_image(source_path: Path, output_paths: dict[str, Path], config: dict) -> str:
    image = read_image(source_path)
    rotation_degrees = 0
    quad, method = detect_photo_quad(
        image,
        keep_white_border=bool(config.get("keep_white_border", False)),
        accept_already_cropped=bool(config.get("accept_already_cropped", False)),
    )
    if _should_use_gemini(config, method):
        gemini_quad, gemini_method, gemini_rotation = detect_photo_quad_with_gemini(source_path, image.shape, config)
        if gemini_quad is not None:
            quad, method = gemini_quad, gemini_method
            rotation_degrees = gemini_rotation
        else:
            logging.info("Gemini no reemplazo la deteccion local para %s (%s)", source_path.name, gemini_method)

    if config.get("debug", True):
        debug_image = draw_debug_contour(image, quad, method)
        debug_path = output_paths["debug"] / safe_output_name(source_path)
        save_image(debug_path, debug_image, int(config.get("output_quality", 95)))

    if quad is None:
        review_path = copy_to_needs_review(source_path, output_paths["needs_review"])
        logging.warning("No se detecto un recorte confiable: %s -> %s", source_path.name, review_path)
        return "needs_review"

    warped = warp_perspective(
        image,
        quad,
        margin_percent=float(config.get("margin_percent", 1.0)),
        points_are_ordered=method.startswith("gemini:"),
    )
    warped = rotate_by_degrees(warped, rotation_degrees)
    if _should_check_orientation_with_gemini(config):
        extra_rotation, orientation_method = detect_upright_rotation_with_gemini(warped, config)
        if extra_rotation:
            warped = rotate_by_degrees(warped, extra_rotation)
            method = f"{method} {orientation_method} rot{extra_rotation}"
    if config.get("rotate_to_landscape", True):
        warped = rotate_to_landscape_if_close(warped)
    corrected = correct_color(warped, config)

    cropped_path = output_paths["cropped"] / safe_output_name(source_path)
    save_image(cropped_path, corrected, int(config.get("output_quality", 95)))
    logging.info("Procesada: %s (%s)", source_path.name, method)
    return "cropped"


def _should_use_gemini(config: dict, method: str) -> bool:
    gemini_config = config.get("gemini", {})
    if not gemini_config.get("enabled", False):
        return False

    mode = str(gemini_config.get("mode", "fallback")).lower()
    if mode == "always":
        return True
    if mode != "fallback":
        return False

    fallback_methods = set(gemini_config.get("fallback_methods", []))
    return method in fallback_methods


def _should_check_orientation_with_gemini(config: dict) -> bool:
    gemini_config = config.get("gemini", {})
    return bool(gemini_config.get("enabled", False) and gemini_config.get("orientation_check", True))


def main() -> int:
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    output_paths = ensure_output_dirs(args.output)

    try:
        image_files = iter_image_files(args.input)
    except FileNotFoundError as error:
        logging.error(str(error))
        return 1

    if not image_files:
        logging.warning("No se encontraron imagenes JPG, JPEG, PNG o WEBP en %s", args.input)
        return 0

    counts = {"cropped": 0, "needs_review": 0, "errors": 0}
    for source_path in tqdm(image_files, desc="Procesando", unit="img"):
        try:
            result = process_image(source_path, output_paths, config)
            counts[result] += 1
        except Exception as error:
            counts["errors"] += 1
            logging.exception("Error procesando %s: %s", source_path, error)
            try:
                copy_to_needs_review(source_path, output_paths["needs_review"])
            except Exception:
                logging.exception("No se pudo copiar %s a needs_review", source_path)

    logging.info(
        "Finalizado. Recortadas: %s | Para revisar: %s | Errores: %s",
        counts["cropped"],
        counts["needs_review"],
        counts["errors"],
    )
    return 0 if counts["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
