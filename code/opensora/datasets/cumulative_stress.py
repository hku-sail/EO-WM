"""Daily cumulative weather-stress features used by EO-WM."""
from __future__ import annotations
import numpy as np

CUMSTRESS_VARIANTS = frozenset({'directional_product'})
RR_IDX = 0
PP_IDX = 1
TG_IDX = 2
TN_IDX = 3
TX_IDX = 4


def _cumsum_from(values: np.ndarray, start_index: int) -> np.ndarray:
    """Cumulative sum reset to zero immediately before ``start_index``."""
    cumulative = np.cumsum(values, axis=0, dtype=np.float32)
    if start_index > 0:
        cumulative = cumulative - cumulative[start_index - 1]
        cumulative[:start_index] = 0.0
    return cumulative.astype(np.float32)


def _valid_weighted_daily_mean(dense_values: np.ndarray, dense_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce daily grids to one PP-valid tile value per channel.

    Args:
        dense_values: ``[T,C,H,W]`` or ``[T,H,W]``.
        dense_valid: ``[T,H,W]`` validity derived from PP.

    Returns:
        ``(daily_mean, daily_valid)``. ``daily_mean`` is ``[T,C]`` for a
        channelled input and ``[T]`` otherwise. A day with no valid pixels has
        a zero placeholder and ``daily_valid=False``; it contributes no
        increment to any stress variant.
    """
    values = np.asarray(dense_values, dtype=np.float32)
    valid = np.asarray(dense_valid, dtype=np.float32)
    squeeze_channel = values.ndim == 3
    if squeeze_channel:
        values = values[:, None]
    if values.ndim != 4:
        raise ValueError(f'dense_values must have shape [T,C,H,W] or [T,H,W], got {values.shape}')
    if valid.shape != (values.shape[0], values.shape[2], values.shape[3]):
        raise ValueError(
            f'dense_valid must have shape {(values.shape[0], values.shape[2], values.shape[3])}, got {valid.shape}')
    weights = (np.isfinite(valid) & (valid > 0.0)).astype(np.float32)[:, None]
    finite_values = np.where(np.isfinite(values), values, 0.0)
    numerator = (finite_values * weights).sum(axis=(-2, -1), dtype=np.float64)
    denominator = weights.sum(axis=(-2, -1), dtype=np.float64)
    daily_mean = np.divide(numerator,
                           denominator,
                           out=np.zeros_like(numerator, dtype=np.float64),
                           where=denominator > 0).astype(np.float32)
    daily_valid = (denominator[:, 0] > 0).astype(np.float32)
    if squeeze_channel:
        daily_mean = daily_mean[:, 0]
    return (daily_mean, daily_valid)


def build_cumulative_stress_features(meso_full: np.ndarray,
                                     meso_indices: np.ndarray,
                                     valid_mask: np.ndarray,
                                     *,
                                     temp_mean_anomaly: np.ndarray | None = None,
                                     precip_anomaly: np.ndarray | None = None,
                                     accumulation_start_index: int | None = None,
                                     log_compress: bool = True) -> np.ndarray:
    """Build cumulative heat, water-deficit, and compound stress from daily weather anomalies."""
    meso = np.asarray(meso_full, dtype=np.float32)
    if meso.ndim != 4 or meso.shape[1] < 5:
        raise ValueError(f'meso_full must have shape [T,5,H,W], got {meso.shape}')
    t_meso, _, height, width = meso.shape
    if t_meso <= 0:
        raise ValueError('meso_full must contain at least one dense weather day')
    valid = np.asarray(valid_mask, dtype=np.float32)
    if valid.shape != (t_meso, height, width):
        raise ValueError(f'valid_mask must have shape {(t_meso, height, width)}, got {valid.shape}')
    valid = (np.isfinite(valid) & (valid > 0.0)).astype(np.float32)
    output_indices = np.clip(np.asarray(meso_indices, dtype=np.int64), 0, max(t_meso - 1, 0))
    if accumulation_start_index is None:
        raise ValueError(
            'accumulation_start_index is required and must be computed from first_highres_index * T_meso / T_highres')
    start_index = int(np.clip(accumulation_start_index, 0, max(t_meso - 1, 0)))
    tile_meso, daily_valid = _valid_weighted_daily_mean(meso, valid)
    tile_temp_z = None
    if temp_mean_anomaly is not None:
        temp_z = np.asarray(temp_mean_anomaly, dtype=np.float32)
        if temp_z.shape != valid.shape:
            raise ValueError(f'temp_mean_anomaly must have shape {valid.shape}, got {temp_z.shape}')
        tile_temp_z, _ = _valid_weighted_daily_mean(temp_z, valid)
    tile_precip_z = None
    if precip_anomaly is not None:
        precip_z = np.asarray(precip_anomaly, dtype=np.float32)
        if precip_z.shape != valid.shape:
            raise ValueError(f'precip_anomaly must have shape {valid.shape}, got {precip_z.shape}')
        tile_precip_z, _ = _valid_weighted_daily_mean(precip_z, valid)
    if tile_temp_z is None or tile_precip_z is None:
        raise ValueError(f"{'directional_product'} requires standardized temperature and precipitation anomalies")
    first = _cumsum_from(np.maximum(tile_temp_z, 0.0) * daily_valid, start_index)
    second = _cumsum_from(np.maximum(-tile_precip_z, 0.0) * daily_valid, start_index)
    third = first * second
    dense_features = np.stack([first, second, third], axis=1).astype(np.float32)
    features = dense_features[output_indices].copy()
    if log_compress:
        features = np.log1p(np.maximum(features, 0.0))
    if features.shape != (len(output_indices), 3):
        raise AssertionError(f'Unexpected stress feature shape: {features.shape}')
    if not np.isfinite(features).all():
        raise ValueError('Stress features contain non-finite values')
    return features.astype(np.float32)
