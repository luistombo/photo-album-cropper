from __future__ import annotations

import cv2
import numpy as np

from src.utils import clamp


def gray_world_white_balance(image: np.ndarray, strength: float) -> np.ndarray:
    strength = clamp(strength, 0.0, 1.0)
    if strength == 0:
        return image

    image_float = image.astype(np.float32)
    means = image_float.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
    gains = gray / np.maximum(means, 1.0)
    gains = 1.0 + (gains - 1.0) * strength
    balanced = image_float * gains
    return np.clip(balanced, 0, 255).astype(np.uint8)


def lab_autocontrast(image: np.ndarray, strength: float) -> np.ndarray:
    strength = clamp(strength, 0.0, 1.0)
    if strength == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)

    low, high = np.percentile(lightness, (1.0, 99.0))
    if high - low < 10:
        return image

    stretched = (lightness.astype(np.float32) - low) * (255.0 / (high - low))
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    mixed = cv2.addWeighted(lightness, 1.0 - strength, stretched, strength, 0)

    corrected_lab = cv2.merge([mixed, a_channel, b_channel])
    return cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)


def apply_gamma(image: np.ndarray, gamma: float, strength: float) -> np.ndarray:
    if abs(gamma - 1.0) < 0.01 or strength <= 0:
        return image

    gamma = clamp(gamma, 0.7, 1.4)
    blended_gamma = 1.0 + (gamma - 1.0) * clamp(strength, 0.0, 1.0)
    inv_gamma = 1.0 / blended_gamma
    table = np.array([((value / 255.0) ** inv_gamma) * 255 for value in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def adjust_saturation(image: np.ndarray, saturation: float, strength: float) -> np.ndarray:
    if abs(saturation - 1.0) < 0.01 or strength <= 0:
        return image

    saturation = clamp(saturation, 0.8, 1.25)
    blended_saturation = 1.0 + (saturation - 1.0) * clamp(strength, 0.0, 1.0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= blended_saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 245)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def protect_highlights(original: np.ndarray, corrected: np.ndarray, threshold: int = 238) -> np.ndarray:
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    mask = original_gray > threshold
    if not np.any(mask):
        return corrected

    result = corrected.copy()
    result[mask] = cv2.addWeighted(original, 0.65, corrected, 0.35, 0)[mask]
    return result


def percentile_white_balance(image: np.ndarray, strength: float) -> np.ndarray:
    strength = clamp(strength, 0.0, 1.0)
    if strength == 0:
        return image

    image_float = image.astype(np.float32)
    balanced = image_float.copy()
    for channel in range(3):
        low, high = np.percentile(image_float[:, :, channel], (1.0, 99.2))
        if high - low < 20:
            continue
        stretched = (image_float[:, :, channel] - low) * (255.0 / (high - low))
        balanced[:, :, channel] = np.clip(stretched, 0, 255)

    mixed = cv2.addWeighted(image_float, 1.0 - strength, balanced, strength, 0)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def neutralize_age_cast_lab(image: np.ndarray, strength: float) -> np.ndarray:
    strength = clamp(strength, 0.0, 1.0)
    if strength == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness, a_channel, b_channel = cv2.split(lab)
    midtones = (lightness > 35) & (lightness < 235)
    if np.count_nonzero(midtones) < image.shape[0] * image.shape[1] * 0.1:
        return image

    a_shift = float(np.median(a_channel[midtones]) - 128.0)
    b_shift = float(np.median(b_channel[midtones]) - 128.0)
    a_channel -= a_shift * strength
    b_channel -= b_shift * strength

    corrected_lab = cv2.merge(
        [
            lightness,
            np.clip(a_channel, 0, 255),
            np.clip(b_channel, 0, 255),
        ]
    ).astype(np.uint8)
    return cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)


def clahe_local_contrast(image: np.ndarray, strength: float) -> np.ndarray:
    strength = clamp(strength, 0.0, 1.0)
    if strength == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(lightness)
    mixed = cv2.addWeighted(lightness, 1.0 - strength, enhanced, strength, 0)
    return cv2.cvtColor(cv2.merge([mixed, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def recover_shadows(image: np.ndarray, strength: float) -> np.ndarray:
    strength = clamp(strength, 0.0, 1.0)
    if strength == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness_float = lightness.astype(np.float32)
    lifted = 255.0 * np.power(lightness_float / 255.0, 0.82)
    shadow_weight = np.clip((150.0 - lightness_float) / 150.0, 0.0, 1.0) * strength
    recovered = lightness_float * (1.0 - shadow_weight) + lifted * shadow_weight
    recovered = np.clip(recovered, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([recovered, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def adjust_vibrance(image: np.ndarray, amount: float, strength: float) -> np.ndarray:
    if abs(amount - 1.0) < 0.01 or strength <= 0:
        return image

    amount = clamp(amount, 0.8, 1.35)
    strength = clamp(strength, 0.0, 1.0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation = hsv[:, :, 1]
    low_saturation_weight = 1.0 - np.clip(saturation / 180.0, 0.0, 1.0)
    hsv[:, :, 1] *= 1.0 + (amount - 1.0) * low_saturation_weight * strength
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 245)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def reduce_specular_reflections(image: np.ndarray, strength: float) -> np.ndarray:
    strength = clamp(strength, 0.0, 1.0)
    if strength == 0:
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    saturation = hsv[:, :, 1]
    local_background = cv2.GaussianBlur(value, (0, 0), sigmaX=13, sigmaY=13)
    local_glare = value - local_background

    mask = ((value > 210) & (saturation < 95) & (local_glare > 16)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

    mask_area = float(np.count_nonzero(mask)) / float(mask.size)
    if mask_area <= 0.0005 or mask_area > 0.10:
        return image

    inpainted = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    feather = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=3, sigmaY=3)
    feather = np.clip(feather * strength, 0.0, 1.0)[:, :, None]
    blended = image.astype(np.float32) * (1.0 - feather) + inpainted.astype(np.float32) * feather
    return np.clip(blended, 0, 255).astype(np.uint8)


def restore_old_paper_photo(image: np.ndarray, config: dict, strength: float) -> np.ndarray:
    restoration_config = config.get("restoration", {})
    if not restoration_config.get("enabled", True):
        return image

    restored = image
    restored = reduce_specular_reflections(
        restored,
        float(restoration_config.get("glare_reduction_strength", 0.35)) * strength,
    )
    restored = percentile_white_balance(
        restored,
        float(restoration_config.get("white_balance_strength", 0.45)) * strength,
    )
    restored = neutralize_age_cast_lab(
        restored,
        float(restoration_config.get("age_cast_reduction_strength", 0.55)) * strength,
    )
    restored = clahe_local_contrast(
        restored,
        float(restoration_config.get("local_contrast_strength", 0.45)) * strength,
    )
    restored = recover_shadows(
        restored,
        float(restoration_config.get("shadow_recovery_strength", 0.28)) * strength,
    )
    restored = adjust_vibrance(
        restored,
        float(restoration_config.get("vibrance", 1.12)),
        strength,
    )
    return restored


def correct_color(image: np.ndarray, config: dict) -> np.ndarray:
    strength = clamp(float(config.get("color_correction_strength", 0.6)), 0.0, 1.0)
    if strength == 0:
        return image

    original = image.copy()
    corrected = restore_old_paper_photo(image, config, strength)
    corrected = gray_world_white_balance(corrected, strength * 0.25)
    corrected = lab_autocontrast(corrected, strength * 0.25)
    corrected = apply_gamma(corrected, float(config.get("gamma", 1.0)), strength)
    corrected = adjust_saturation(corrected, float(config.get("saturation", 1.05)), strength)
    return protect_highlights(original, corrected)
