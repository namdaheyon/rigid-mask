#!/usr/bin/env python3
"""Offline checks of saved VCN flow and foreground decomposition.

LK, photometric error, and two-view triangulation are consistency checks, not
ground truth. Frozen final-logit attribution does not rerun fgnet.
"""
import argparse
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
import yaml


LEGACY_CONTRIBUTION_NAMES = [
    'symmetric_transfer', 'epipolar', 'angular_2d', 'distance_3d',
    'angular_3d', 'depth_contrast',
]
REPLAY_MAE_LIMIT = 1e-5
REPLAY_MAX_ERROR_LIMIT = 1e-4
MIN_BASELINE_M = 0.05
MIN_PARALLAX_RAD = 0.001
MIN_ACCEPTED_RATIO = 0.15
MIN_ACCEPTED_TRIANGULATIONS = 8


def capture_metadata(data):
    if 'metadata_json' not in data:
        return {}
    try:
        value = json.loads(str(data['metadata_json']))
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def original_roi_mask(data, roi, target_shape):
    """Map a full-frame [x0,y0,x1,y1) rectangle to any target H,W grid."""
    target_h, target_w = map(int, target_shape)
    if target_h <= 0 or target_w <= 0:
        raise ValueError('ROI target resolution must be positive')
    if 'previous_bgr' not in data or np.asarray(data['previous_bgr']).ndim < 2:
        raise ValueError('previous_bgr is required to interpret full-frame ROI coordinates')
    original_h, original_w = np.asarray(data['previous_bgr']).shape[:2]
    if roi is None:
        return np.ones((target_h, target_w), dtype=bool)
    x0, y0, x1, y1 = map(int, roi)
    if not (0 <= x0 < x1 <= original_w and 0 <= y0 < y1 <= original_h):
        raise ValueError(
            'ROI must satisfy 0 <= x0 < x1 <= {} and 0 <= y0 < y1 <= {}'.format(
                original_w, original_h))
    full_frame = np.zeros((original_h, original_w), dtype=np.uint8)
    full_frame[y0:y1, x0:x1] = 1
    if (target_h, target_w) == (original_h, original_w):
        return full_frame.astype(bool)
    return cv2.resize(
        full_frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(bool)


def percentiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return dict(
        count=int(values.size),
        p05_p50_p95=np.percentile(values, [5, 50, 95]).tolist()
        if values.size else None)


def _single_spatial_map(value, name):
    array = np.asarray(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError('{} must have only singleton batch/channel axes; got {}'.format(
            name, list(np.asarray(value).shape)))
    return array


def contribution_metadata(data, metadata):
    result = dict(available='foreground_contributions' in data, warnings=[], errors=[])
    if 'foreground_contributions' not in data:
        result['errors'].append('foreground_contributions missing')
        return result
    tensor = np.asarray(data['foreground_contributions'])
    result.update(shape=list(tensor.shape), dtype=str(tensor.dtype), ndim=tensor.ndim,
                  layout='BCHW' if tensor.ndim == 4 else 'unknown')
    if tensor.ndim == 4:
        result.update(
            batch_dimension=dict(axis=0, size=int(tensor.shape[0])),
            channel_dimension=dict(axis=1, count=int(tensor.shape[1])),
            spatial_resolution=dict(height=int(tensor.shape[2]), width=int(tensor.shape[3])))
        if tensor.shape[0] != 1:
            result['warnings'].append(
                'expected runtime batch size 1, captured {}'.format(tensor.shape[0]))
    else:
        result['errors'].append(
            'foreground_contributions must be BCHW (4 dimensions)')

    if 'foreground_contribution_channel_names' in metadata:
        names = metadata['foreground_contribution_channel_names']
        source = 'capture_metadata.foreground_contribution_channel_names'
    elif 'contribution_order' in metadata:
        names = metadata['contribution_order']
        source = 'capture_metadata.contribution_order'
    else:
        names = LEGACY_CONTRIBUTION_NAMES
        source = 'fallback_legacy_contribution_order'
        result['warnings'].append('contribution channel names absent from capture metadata')
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        result['errors'].append('saved contribution channel names are not a string list')
        names = []
    result['channel_names'] = names
    result['channel_names_source'] = source
    count = int(tensor.shape[1]) if tensor.ndim == 4 else None
    if count is not None and len(names) != count:
        result['errors'].append(
            'contribution channel-name count {} does not match tensor channel count {}'.format(
                len(names), count))
        result['resolved_channel_names'] = [
            'channel_{:03d}_name_unresolved'.format(index) for index in range(count)]
    else:
        result['resolved_channel_names'] = list(names)
    saved_tensor = metadata.get('tensor_metadata', {}).get(
        'foreground_contributions', {})
    if isinstance(saved_tensor, dict):
        saved_shape = saved_tensor.get('shape')
        saved_dtype = saved_tensor.get('dtype')
        if saved_shape is not None and list(saved_shape) != list(tensor.shape):
            result['errors'].append(
                'saved contribution shape {} does not match actual {}'.format(
                    saved_shape, list(tensor.shape)))
        if saved_dtype is not None and saved_dtype != str(tensor.dtype):
            result['errors'].append(
                'saved contribution dtype {} does not match actual {}'.format(
                    saved_dtype, tensor.dtype))
    return result


def fgnet_input_metadata(data, metadata):
    result = dict(available='fgnet_input_tensor' in data, warnings=[], errors=[])
    if 'fgnet_input_tensor' not in data:
        result['reason'] = 'fgnet input tensor missing from legacy capture'
        return result
    tensor = np.asarray(data['fgnet_input_tensor'])
    result.update(shape=list(tensor.shape), dtype=str(tensor.dtype), ndim=tensor.ndim,
                  layout='BCHW' if tensor.ndim == 4 else 'unknown')
    if tensor.ndim != 4:
        result['errors'].append('fgnet_input_tensor must be BCHW (4 dimensions)')
        return result
    result.update(
        batch_dimension=dict(axis=0, size=int(tensor.shape[0])),
        channel_dimension=dict(axis=1, count=int(tensor.shape[1])),
        spatial_resolution=dict(height=int(tensor.shape[2]), width=int(tensor.shape[3])))
    names = metadata.get('fgnet_input_channel_names')
    if names is None:
        capture = metadata.get('fgnet_input_capture', {})
        names = capture.get('channel_names') if isinstance(capture, dict) else None
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        result['errors'].append('fgnet input channel names missing or invalid')
        result['channel_names'] = []
    else:
        result['channel_names'] = names
        if len(names) != tensor.shape[1]:
            result['errors'].append(
                'fgnet input channel-name count {} does not match tensor channel count {}'.format(
                    len(names), tensor.shape[1]))
    saved_tensor = metadata.get('tensor_metadata', {}).get('fgnet_input_tensor', {})
    if isinstance(saved_tensor, dict):
        if (saved_tensor.get('shape') is not None
                and list(saved_tensor['shape']) != list(tensor.shape)):
            result['errors'].append('saved fgnet input shape does not match actual tensor')
        if (saved_tensor.get('dtype') is not None
                and saved_tensor['dtype'] != str(tensor.dtype)):
            result['errors'].append('saved fgnet input dtype does not match actual tensor')
    return result


def runtime_threshold(metadata):
    saved = metadata.get('foreground_thresholds')
    if isinstance(saved, dict):
        value = saved.get('dynamic_probability_threshold')
        if isinstance(value, (int, float)) and np.isfinite(value) and 0 <= value <= 1:
            return dict(
                value=float(value), source='capture_metadata.foreground_thresholds',
                base_threshold=saved.get('base_threshold'),
                probability_margin=saved.get('probability_margin'),
                static_probability_threshold=saved.get('static_probability_threshold'))
    return dict(
        value=0.6, source='fallback_legacy_0.6', base_threshold=None,
        probability_margin=None, static_probability_threshold=None,
        warning='Actual runtime threshold unavailable in this legacy capture')


def _native_reconstruction(data):
    import torch

    terms = np.array(data['foreground_contributions'], copy=True)
    residual = np.array(data['foreground_residual'], copy=True)
    if terms.ndim != 4:
        raise ValueError('foreground_contributions must be BCHW')
    if residual.ndim != 4 or residual.shape[0] != terms.shape[0] or residual.shape[1] != 1 \
            or residual.shape[2:] != terms.shape[2:]:
        raise ValueError(
            'foreground_residual must be B1HW and match contributions; got {} vs {}'.format(
                list(residual.shape), list(terms.shape)))
    terms_tensor = torch.from_numpy(terms)
    residual_tensor = torch.from_numpy(residual)
    return terms_tensor, residual_tensor, terms_tensor.sum(1, keepdim=True) + residual_tensor


def _resize_logit(logit, target_shape):
    import torch.nn.functional as functional

    if tuple(logit.shape[-2:]) == tuple(target_shape):
        return logit
    return functional.interpolate(
        logit, tuple(target_shape), mode='bilinear', align_corners=False)


def _error_metrics(actual, reconstructed, selection):
    valid = selection & np.isfinite(actual) & np.isfinite(reconstructed)
    difference = np.abs(actual[valid] - reconstructed[valid])
    return dict(
        comparison_pixel_count=int(difference.size),
        mae=float(difference.mean()) if difference.size else None,
        max_absolute_error=float(difference.max()) if difference.size else None)


def replay_validation(data, roi, threshold):
    result = dict(
        formula='sum(foreground_contributions, channel) + foreground_residual',
        runtime_order=[
            'native final logit',
            'torch bilinear resize to network input resolution (align_corners=False)',
            'sigmoid',
            'OpenCV INTER_LINEAR resize to original full frame',
        ], warnings=[])
    required = [name for name in ('foreground_contributions', 'foreground_residual')
                if name not in data]
    if required:
        result.update(available=False, sufficient_match=False,
                      reason='missing ' + ', '.join(required))
        result['warnings'].append(
            'Replay validation unavailable; frozen attribution must not be trusted')
        return result
    try:
        _, _, native = _native_reconstruction(data)
    except ValueError as error:
        result.update(available=False, sufficient_match=False, reason=str(error))
        result['warnings'].append(
            'Replay validation failed; frozen attribution must not be trusted')
        return result

    metadata = capture_metadata(data)
    replay_meta = metadata.get('foreground_replay', {})
    logit_key = replay_meta.get('final_logit_key', 'foreground_logits')
    probability_key = replay_meta.get('final_probability_key', 'probability')
    actual_logit = None
    if logit_key in data:
        try:
            actual_logit = _single_spatial_map(data[logit_key], logit_key)
        except ValueError as error:
            result['warnings'].append(str(error))
    if actual_logit is not None:
        network_shape = actual_logit.shape
    elif 'vcn_flow_x_network_px' in data:
        network_shape = np.asarray(data['vcn_flow_x_network_px']).shape[-2:]
    else:
        network_shape = native.shape[-2:]
        result['warnings'].append('Network resolution inferred from contribution grid')

    reconstructed_logit = _resize_logit(native, network_shape)[0, 0].numpy()
    if actual_logit is not None:
        logit_selection = original_roi_mask(data, roi, network_shape)
        result['logit'] = _error_metrics(
            actual_logit, reconstructed_logit, logit_selection)
        result['logit']['actual_key'] = logit_key
    else:
        result['logit'] = dict(available=False, reason='actual final logit unavailable')

    actual_probability = None
    if probability_key in data:
        try:
            actual_probability = _single_spatial_map(data[probability_key], probability_key)
        except ValueError as error:
            result['warnings'].append(str(error))
    if actual_probability is None:
        result.update(available=False, sufficient_match=False)
        result['probability'] = dict(
            available=False, reason='actual final foreground probability unavailable')
        result['warnings'].append(
            'Replay probability cannot be validated; frozen attribution must not be trusted')
        return result

    import torch
    reconstructed_network_probability = torch.sigmoid(
        _resize_logit(native, network_shape))[0, 0].numpy()
    probability_shape = actual_probability.shape
    if tuple(probability_shape) == tuple(network_shape):
        reconstructed_probability = reconstructed_network_probability
    else:
        reconstructed_probability = cv2.resize(
            reconstructed_network_probability,
            (probability_shape[1], probability_shape[0]),
            interpolation=cv2.INTER_LINEAR)
    probability_selection = original_roi_mask(data, roi, probability_shape)
    probability_metrics = _error_metrics(
        actual_probability, reconstructed_probability, probability_selection)
    valid = (probability_selection & np.isfinite(actual_probability)
             & np.isfinite(reconstructed_probability))
    if valid.any():
        actual_binary = actual_probability[valid] >= threshold['value']
        reconstructed_binary = reconstructed_probability[valid] >= threshold['value']
        disagreement = float(np.mean(actual_binary != reconstructed_binary))
    else:
        disagreement = None
    probability_metrics.update(
        actual_key=probability_key,
        binary_mask_disagreement_fraction=disagreement,
        dynamic_threshold=threshold)
    result['probability'] = probability_metrics

    logit_ok = True
    if actual_logit is not None:
        logit_ok = (
            result['logit']['comparison_pixel_count'] > 0
            and result['logit']['mae'] <= REPLAY_MAE_LIMIT
            and result['logit']['max_absolute_error'] <= REPLAY_MAX_ERROR_LIMIT)
    probability_ok = (
        probability_metrics['comparison_pixel_count'] > 0
        and probability_metrics['mae'] <= REPLAY_MAE_LIMIT
        and probability_metrics['max_absolute_error'] <= REPLAY_MAX_ERROR_LIMIT
        and disagreement == 0.0)
    result.update(
        available=True, sufficient_match=bool(logit_ok and probability_ok),
        tolerances=dict(mae=REPLAY_MAE_LIMIT,
                        max_absolute_error=REPLAY_MAX_ERROR_LIMIT,
                        binary_mask_disagreement_fraction=0.0))
    if not result['sufficient_match']:
        result['warnings'].append(
            'Reconstruction does not sufficiently match runtime output; '
            'subsequent frozen attribution must not be trusted')
    return result


def frozen_final_logit_attribution(data, roi, contribution_info, threshold, replay):
    warning = 'This is NOT true input ablation and NOT fgnet re-inference'
    result = dict(
        method='frozen_final_logit_attribution', warning=warning,
        explanation='Saved final additive terms are removed without rerunning the CNN',
        trusted=bool(replay.get('sufficient_match')), variants={})
    if not result['trusted']:
        result['trust_warning'] = (
            'Replay validation is unavailable or insufficient; do not trust these values')
    if ('foreground_contributions' not in data or
            'foreground_residual' not in data):
        result['reason'] = 'spatial contributions or residual missing'
        return result
    try:
        terms, residual, original = _native_reconstruction(data)
    except ValueError as error:
        result['reason'] = str(error)
        return result

    if 'foreground_logits' in data:
        target_shape = _single_spatial_map(
            data['foreground_logits'], 'foreground_logits').shape
    elif 'vcn_flow_x_network_px' in data:
        target_shape = np.asarray(data['vcn_flow_x_network_px']).shape[-2:]
    else:
        target_shape = terms.shape[-2:]
    if 'probability' in data:
        output_shape = _single_spatial_map(data['probability'], 'probability').shape
    else:
        output_shape = np.asarray(data['gated_mask']).shape
    selection = original_roi_mask(data, roi, output_shape)
    if roi is None and 'gated_mask' in data:
        gated = cv2.resize(
            np.asarray(data['gated_mask']), (output_shape[1], output_shape[0]),
            interpolation=cv2.INTER_NEAREST)
        selection &= gated != 127

    variants = {'original_reconstructed': original,
                'without_residual': terms.sum(1, keepdim=True)}
    names = contribution_info.get('resolved_channel_names', [])
    for index in range(terms.shape[1]):
        name = names[index] if index < len(names) else 'channel_{:03d}'.format(index)
        variants['without_' + name] = original - terms[:, index:index + 1]

    import torch
    for name, logits in variants.items():
        probability = torch.sigmoid(_resize_logit(logits, target_shape))[0, 0].numpy()
        if tuple(output_shape) != tuple(target_shape):
            probability = cv2.resize(
                probability, (output_shape[1], output_shape[0]),
                interpolation=cv2.INTER_LINEAR)
        values = probability[selection]
        result['variants'][name] = dict(
            pixels=int(values.size), probability=percentiles(values),
            dynamic_fraction=float(np.mean(values >= threshold['value']))
            if values.size else None)
    result['dynamic_threshold'] = threshold
    result['selection'] = 'manual full-frame ROI' if roi else 'LiDAR-supported confident pixels'
    return result


def tau_convention_audit(data, roi):
    raw_key = 'optical_expansion_tau_Z1_over_Z0'
    converted_key = 'depth_ratio_Z0_over_Z1'
    result = dict(
        raw_key=raw_key, raw_convention='Z1/Z0', converted_key=converted_key,
        converted_convention='Z0/Z1', expected_product=1.0)
    if raw_key not in data or converted_key not in data:
        result.update(
            available=False,
            reason='raw tau missing from legacy capture' if raw_key not in data
            else 'converted depth ratio missing')
        return result
    try:
        raw = _single_spatial_map(data[raw_key], raw_key)
        converted = _single_spatial_map(data[converted_key], converted_key)
    except ValueError as error:
        result.update(available=False, reason=str(error))
        return result
    if raw.shape != converted.shape:
        result.update(
            available=False,
            reason='tau grids differ: {} vs {}'.format(list(raw.shape), list(converted.shape)))
        return result
    selection = original_roi_mask(data, roi, raw.shape)
    valid = (selection & np.isfinite(raw) & np.isfinite(converted)
             & (raw > 0) & (converted > 0))
    error = np.abs(raw[valid] * converted[valid] - 1.0)
    result.update(
        available=True, valid_pixel_count=int(error.size),
        absolute_product_minus_one=percentiles(error),
        reciprocal_consistent=bool(error.size and np.percentile(error, 95) <= 1e-5))
    return result


def yaw_rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def camera_motion(data, lidar_origin, width, height):
    metadata = capture_metadata(data)
    pair = metadata.get('ego_pair', {})
    if pair.get('reason') != 'valid':
        raise ValueError('vehicle pose not bracketed')
    first, last = pair['previous_xy_yaw'], pair['current_xy_yaw']
    if first[2] is None or last[2] is None:
        raise ValueError('vehicle yaw missing')
    first, last = np.asarray(first, dtype=float), np.asarray(last, dtype=float)
    if not np.isfinite([first, last]).all():
        raise ValueError('nonfinite vehicle pose')
    delta_xy = last[:2] - first[:2]
    baseline = float(np.linalg.norm(delta_xy))
    delta_yaw = float(np.arctan2(
        np.sin(last[2] - first[2]), np.cos(last[2] - first[2])))
    calibration = yaml.safe_load(str(data['calibration_yaml']))
    camera = calibration['cameras'][str(metadata['camera_id'])]
    rotation = np.asarray(camera['rotation_matrix'], dtype=float)
    translation = np.asarray(camera['translation_vector'], dtype=float)
    offset = np.asarray(lidar_origin, dtype=float) - rotation.T @ translation
    rotation_01 = rotation @ yaw_rotation(delta_yaw) @ rotation.T
    displacement = rotation @ (
        yaw_rotation(first[2]).T @ np.r_[delta_xy, 0.]
        + (yaw_rotation(delta_yaw) - np.eye(3)) @ offset)
    rotation_10 = rotation_01.T
    translation_10 = -rotation_10 @ displacement
    camera_baseline = float(np.linalg.norm(translation_10))
    matrix = np.asarray(data['camera_matrix'], dtype=float).copy()
    original_h, original_w = np.asarray(data['previous_bgr']).shape[:2]
    matrix[0] *= width / original_w
    matrix[1] *= height / original_h
    motion = dict(
        vehicle_translation_baseline_m=baseline,
        camera_translation_norm_m=camera_baseline,
        yaw_change_rad=delta_yaw,
        yaw_change_deg=float(np.degrees(delta_yaw)))
    return matrix, rotation_10, translation_10, motion


def static_depth_check(points0, points1, cnn_ratio, camera_matrix,
                       rotation_10, translation_10, motion):
    """Two-view static-scene triangulation diagnostic, never metric depth truth."""
    count = len(points0)
    base = dict(
        vehicle_translation_baseline_m=motion['vehicle_translation_baseline_m'],
        camera_translation_norm_m=motion['camera_translation_norm_m'],
        yaw_change_rad=motion['yaw_change_rad'],
        yaw_change_deg=motion['yaw_change_deg'],
        lk_match_count=int(count), accepted_triangulation_count=0,
        accepted_ratio=0.0, parallax_rad=percentiles([]),
        reprojection_error_px=percentiles([]), ill_conditioned=True,
        ill_conditioned_reasons=[],
        warning='Triangulated ratio assumes a static ROI and is not metric depth ground truth')
    if count == 0:
        base['ill_conditioned_reasons'].append('no_reliable_LK_matches')
        return base
    inverse = np.linalg.inv(camera_matrix)
    rays0 = np.c_[points0, np.ones(count)] @ inverse.T
    rays1 = np.c_[points1, np.ones(count)] @ inverse.T
    projection0 = np.c_[np.eye(3), np.zeros(3)]
    projection1 = np.c_[rotation_10, translation_10]
    homogeneous = cv2.triangulatePoints(
        projection0, projection1, rays0[:, :2].T, rays1[:, :2].T)
    safe = np.abs(homogeneous[3]) > 1e-10
    xyz0 = np.full((count, 3), np.nan)
    xyz0[safe] = (homogeneous[:3, safe] / homogeneous[3, safe]).T
    xyz1 = xyz0 @ rotation_10.T + translation_10
    with np.errstate(divide='ignore', invalid='ignore'):
        uv0 = xyz0 @ camera_matrix.T
        uv0 = uv0[:, :2] / uv0[:, 2:]
        uv1 = xyz1 @ camera_matrix.T
        uv1 = uv1[:, :2] / uv1[:, 2:]
        ratio = xyz0[:, 2] / xyz1[:, 2]
    reprojection = np.maximum(
        np.linalg.norm(uv0 - points0, axis=1),
        np.linalg.norm(uv1 - points1, axis=1))
    unit0 = rays0 / np.linalg.norm(rays0, axis=1, keepdims=True)
    unit1 = rays1 @ rotation_10
    unit1 /= np.linalg.norm(unit1, axis=1, keepdims=True)
    parallax = np.arctan2(
        np.linalg.norm(np.cross(unit0, unit1), axis=1),
        np.sum(unit0 * unit1, axis=1))
    finite_geometry = safe & np.isfinite(ratio) & np.isfinite(reprojection) & np.isfinite(parallax)
    good = (finite_geometry & np.isfinite(cnn_ratio) & (cnn_ratio > 0)
            & (xyz0[:, 2] > .1) & (xyz1[:, 2] > .1)
            & (reprojection <= 1.) & (parallax >= MIN_PARALLAX_RAD))
    accepted = int(good.sum())
    accepted_ratio = float(accepted / count)
    finite_parallax = parallax[finite_geometry]
    reasons = []
    if motion['vehicle_translation_baseline_m'] < MIN_BASELINE_M:
        reasons.append('vehicle_baseline_below_0.05_m')
    if motion['camera_translation_norm_m'] < MIN_BASELINE_M:
        reasons.append('camera_translation_below_0.05_m')
    if not finite_parallax.size:
        reasons.append('no_finite_parallax')
    elif float(np.median(finite_parallax)) < MIN_PARALLAX_RAD:
        reasons.append('median_parallax_below_{:.6f}_rad'.format(MIN_PARALLAX_RAD))
    if accepted < MIN_ACCEPTED_TRIANGULATIONS:
        reasons.append('accepted_triangulations_below_{}'.format(
            MIN_ACCEPTED_TRIANGULATIONS))
    if accepted_ratio < MIN_ACCEPTED_RATIO:
        reasons.append('accepted_ratio_below_{:.2f}'.format(MIN_ACCEPTED_RATIO))
    base.update(
        accepted_triangulation_count=accepted,
        accepted_ratio=accepted_ratio,
        parallax_rad=percentiles(finite_parallax),
        reprojection_error_px=percentiles(reprojection[finite_geometry]),
        min_parallax_rad=MIN_PARALLAX_RAD,
        max_reprojection_px=1.0,
        triangulated_Z0_over_Z1=percentiles(ratio[good]),
        cnn_Z0_over_Z1=percentiles(cnn_ratio[good]),
        absolute_log_ratio_error=percentiles(
            np.abs(np.log(cnn_ratio[good] / ratio[good]))),
        ill_conditioned=bool(reasons), ill_conditioned_reasons=reasons)
    return base


def unavailable_static_depth(reason, motion=None, lk_count=0):
    motion = motion or {}
    return dict(
        vehicle_translation_baseline_m=motion.get('vehicle_translation_baseline_m'),
        camera_translation_norm_m=motion.get('camera_translation_norm_m'),
        yaw_change_rad=motion.get('yaw_change_rad'),
        yaw_change_deg=motion.get('yaw_change_deg'),
        lk_match_count=int(lk_count), accepted_triangulation_count=0,
        accepted_ratio=0.0, parallax_rad=percentiles([]),
        reprojection_error_px=percentiles([]), ill_conditioned=True,
        ill_conditioned_reasons=[reason],
        warning='Triangulated ratio assumes a static ROI and is not metric depth ground truth')


def audit(data, roi=None, assume_static=False, lidar_origin=None):
    if assume_static and (roi is None or lidar_origin is None):
        raise ValueError('Static depth check requires a manually verified ROI and LiDAR origin')
    if lidar_origin is not None and (
            len(lidar_origin) != 3 or not np.isfinite(lidar_origin).all()):
        raise ValueError('LiDAR origin must contain three finite coordinates')
    metadata = capture_metadata(data)
    threshold = runtime_threshold(metadata)
    contribution_info = contribution_metadata(data, metadata)
    fgnet_input_info = fgnet_input_metadata(data, metadata)

    flow_x = np.asarray(data['vcn_flow_x_network_px'])[0]
    flow_y = np.asarray(data['vcn_flow_y_network_px'])[0]
    height, width = flow_x.shape
    previous = cv2.resize(np.asarray(data['previous_bgr']), (width, height))
    current = cv2.resize(np.asarray(data['current_bgr']), (width, height))
    gray0 = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(
        np.asarray(data['gated_mask']), (width, height), interpolation=cv2.INTER_NEAREST)
    roi_mask = original_roi_mask(data, roi, (height, width))
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x, map_y = xx + flow_x, yy + flow_y
    valid = (np.isfinite(map_x) & np.isfinite(map_y) & (map_x >= 0)
             & (map_x < width - 1) & (map_y >= 0) & (map_y < height - 1))
    warped = cv2.remap(
        gray1, np.where(valid, map_x, 0).astype(np.float32),
        np.where(valid, map_y, 0).astype(np.float32), cv2.INTER_LINEAR)
    photometric_error = np.abs(warped.astype(float) - gray0.astype(float))
    depth_ratio = cv2.resize(
        _single_spatial_map(data['depth_ratio_Z0_over_Z1'],
                            'depth_ratio_Z0_over_Z1'),
        (width, height), interpolation=cv2.INTER_LINEAR)
    report = dict(
        units='network pixels; photometric error in grayscale 0..255',
        warning='Consistency checks, not ground truth; D/S are model labels',
        runtime_dynamic_threshold=threshold,
        foreground_contribution_metadata=contribution_info,
        fgnet_input_metadata=fgnet_input_info,
        tau_convention_validation=tau_convention_audit(data, roi), regions={})

    feature_mask = (roi_mask if assume_static else ((mask != 127) & roi_mask)).astype(np.uint8) * 255
    points = cv2.goodFeaturesToTrack(
        gray0, maxCorners=500, qualityLevel=.01, minDistance=5, mask=feature_mask)
    report['lk'] = dict(detected=0 if points is None else len(points), accepted=0)
    motion_solution = None
    motion = None
    if assume_static:
        try:
            motion_solution = camera_motion(data, lidar_origin, width, height)
            motion = motion_solution[3]
        except (ValueError, KeyError, TypeError) as error:
            report['static_depth_check'] = unavailable_static_depth(str(error))
    if points is not None:
        following, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray0, gray1, points, None)
        if following is not None:
            back, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                gray1, gray0, following, None)
            if back is not None:
                first, second = points[:, 0], following[:, 0]
                forward_backward = np.linalg.norm(back[:, 0] - first, axis=1)
                keep = ((forward_status[:, 0] != 0) & (backward_status[:, 0] != 0)
                        & (forward_backward <= 1.))
                keep &= (np.isfinite(second).all(1) & (second[:, 0] >= 0)
                         & (second[:, 0] < width) & (second[:, 1] >= 0)
                         & (second[:, 1] < height))
                sample_x = first[:, 0].reshape(-1, 1)
                sample_y = first[:, 1].reshape(-1, 1)
                cnn_flow = np.column_stack([
                    cv2.remap(component, sample_x, sample_y, cv2.INTER_LINEAR)[:, 0]
                    for component in (flow_x, flow_y)])
                keep &= np.isfinite(cnn_flow).all(1)
                discrepancy = np.linalg.norm(cnn_flow - (second - first), axis=1)
                labels = mask[
                    np.rint(first[:, 1]).astype(int), np.rint(first[:, 0]).astype(int)]
                report['lk'] = dict(
                    detected=len(points), accepted=int(keep.sum()), fb_limit_px=1.,
                    cnn_minus_lk_px=percentiles(discrepancy[keep]),
                    dynamic=percentiles(discrepancy[keep & (labels == 255)]),
                    static=percentiles(discrepancy[keep & (labels == 0)]))
                if assume_static and motion_solution is not None:
                    matrix, rotation_10, translation_10, motion = motion_solution
                    cnn_ratio = cv2.remap(
                        depth_ratio, sample_x, sample_y, cv2.INTER_LINEAR)[:, 0]
                    report['static_depth_check'] = static_depth_check(
                        first[keep], second[keep], cnn_ratio[keep], matrix,
                        rotation_10, translation_10, motion)
    if assume_static and 'static_depth_check' not in report:
        report['static_depth_check'] = unavailable_static_depth(
            'no_reliable_LK_matches', motion, report['lk']['accepted'])

    for name, selection in [('dynamic', mask == 255), ('static', mask == 0)]:
        selected = selection & roi_mask
        usable = selected & valid
        report['regions'][name] = dict(
            pixels=int(selected.sum()),
            flow_endpoint_outside_or_nonfinite=int((selected & ~valid).sum()),
            photometric_error=percentiles(photometric_error[usable]),
            flow_magnitude=percentiles(np.hypot(flow_x, flow_y)[usable]),
            depth_ratio_Z0_over_Z1=percentiles(depth_ratio[selected]))

    replay = replay_validation(data, roi, threshold)
    report['replay_validation'] = replay
    report['frozen_final_logit_attribution'] = frozen_final_logit_attribution(
        data, roi, contribution_info, threshold, replay)
    warnings = []
    warnings.extend(contribution_info.get('warnings', []))
    warnings.extend(contribution_info.get('errors', []))
    warnings.extend(fgnet_input_info.get('warnings', []))
    warnings.extend(fgnet_input_info.get('errors', []))
    warnings.extend(replay.get('warnings', []))
    if threshold.get('warning'):
        warnings.append(threshold['warning'])
    report['warnings'] = warnings
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', nargs='+', type=Path)
    parser.add_argument(
        '--roi', nargs=4, type=int, metavar=('X0', 'Y0', 'X1', 'Y1'),
        help='Optional rectangle in original full-frame coordinates')
    parser.add_argument(
        '--assume-static', action='store_true',
        help='User-verified static ROI; enables triangulated depth-ratio check')
    parser.add_argument(
        '--lidar-origin-vehicle', nargs=3, type=float, metavar=('X', 'Y', 'Z'))
    args = parser.parse_args()
    if len(args.captures) > 30:
        parser.error('At most 30 captures per run')
    output = Path(tempfile.mkdtemp(prefix='rigid_flow_audit_'))
    print('OUTPUT', output, flush=True)
    for index, path in enumerate(args.captures):
        with np.load(path, allow_pickle=False) as data:
            report = audit(
                data, args.roi, args.assume_static, args.lidar_origin_vehicle)
            report.update(
                capture=str(path.resolve()), metadata=capture_metadata(data), roi=args.roi,
                assume_static=args.assume_static,
                lidar_origin_vehicle=args.lidar_origin_vehicle)
        with open(output / '{:02d}.json'.format(index), 'x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        for warning in report.get('warnings', []):
            print('WARNING {}: {}'.format(path, warning), flush=True)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
