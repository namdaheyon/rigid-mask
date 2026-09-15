#!/usr/bin/env python3
"""Offline pose A/B on saved full frames. No ROS publishers or parameter writes.

Uses saved visual pose as baseline to avoid RANSAC randomness. The alternative
is planar vehicle XY/yaw plus calibrated camera extrinsics, NOT 6-DoF truth.
All downstream CNN features, mixing weights and residuals are recomputed.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rigid_mask_ros.model import RigidMaskModel
from rigid_mask_ros.vendor.rigidmask.models.submodule import get_skew_mat


def yaw_rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def vehicle_pose(metadata, calibration, lidar_origin):
    pair = metadata['ego_pair']
    if pair.get('reason') != 'valid':
        raise ValueError('Vehicle pose is not bracketed')
    first, last = pair['previous_xy_yaw'], pair['current_xy_yaw']
    if first[2] is None or last[2] is None:
        raise ValueError('Vehicle yaw missing')
    first, last = np.asarray(first), np.asarray(last)
    if not np.isfinite([first, last]).all():
        raise ValueError('Nonfinite vehicle pose')
    if np.linalg.norm(last[:2] - first[:2]) < .05:
        raise ValueError('Less than 5 cm translation: direction comparison unreliable')
    camera = calibration['cameras'][str(metadata['camera_id'])]
    rotation = np.asarray(camera['rotation_matrix'], dtype=float)
    translation = np.asarray(camera['translation_vector'], dtype=float)
    offset = np.asarray(lidar_origin) - rotation.T @ translation
    delta = last[2] - first[2]
    motion = rotation @ (yaw_rotation(first[2]).T @ np.r_[last[:2]-first[:2], 0.]
                         + (yaw_rotation(delta)-np.eye(3)) @ offset)
    if not np.isfinite(motion).all() or np.linalg.norm(motion) < 1e-6:
        raise ValueError('Invalid camera displacement')
    motion /= np.linalg.norm(motion)
    angle_axis = cv2.Rodrigues(rotation @ yaw_rotation(delta) @ rotation.T)[0]
    rot = angle_axis.reshape(1, 3).astype(np.float32)
    trans = motion.reshape(1, 3).astype(np.float32)
    essential = get_skew_mat(torch.from_numpy(trans), torch.from_numpy(rot)).numpy()
    return dict(rotation=rot, translation=trans, essential=essential)


def runtime_thresholds(metadata):
    saved = metadata.get('foreground_thresholds')
    if isinstance(saved, dict):
        dynamic = saved.get('dynamic_probability_threshold')
        static = saved.get('static_probability_threshold')
        if (isinstance(dynamic, (int, float)) and np.isfinite(dynamic)
                and isinstance(static, (int, float)) and np.isfinite(static)):
            return dict(dynamic=float(dynamic), static=float(static),
                        source='capture_metadata.foreground_thresholds')
    return dict(dynamic=.6, static=.4, source='fallback_legacy_0.6_0.4',
                warning='Actual runtime thresholds unavailable in this legacy capture')


def statistics(probability, selection, thresholds):
    values = probability[selection]
    if not values.size:
        return dict(pixels=0)
    return dict(pixels=int(values.size), mean_probability=float(values.mean()),
                dynamic_fraction=float(np.mean(values >= thresholds['dynamic'])),
                static_fraction=float(np.mean(values <= thresholds['static'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', nargs='+', type=Path)
    parser.add_argument('--lidar-origin-vehicle', nargs=3, type=float, required=True,
                        metavar=('X', 'Y', 'Z'), help='Actual LiDAR mounting position in vehicle frame, metres')
    parser.add_argument('--device', default='cpu', help='cpu or cuda:0; uses existing installation only')
    parser.add_argument('--resolution', type=float, default=.5)
    package = Path(__file__).resolve().parents[1]
    parser.add_argument('--rigidmask-weights', type=Path, default=package/'weights/rigidmask-sf/weights.pth')
    parser.add_argument('--midas-weights', type=Path, default=package/'weights/midas/midas_v21_384.pt')
    args = parser.parse_args()
    if len(args.captures) > 10:
        parser.error('At most 10 captures per run')
    torch.set_num_threads(4)
    output = Path(tempfile.mkdtemp(prefix='rigid_pose_comparison_'))
    print('OUTPUT', output, flush=True)
    model = None
    for index, path in enumerate(args.captures):
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        metadata = json.loads(str(data['metadata_json']))
        thresholds = runtime_thresholds(metadata)
        try:
            alternative = vehicle_pose(metadata, yaml.safe_load(str(data['calibration_yaml'])),
                                       args.lidar_origin_vehicle)
        except ValueError as error:
            print('SKIP', path, str(error), flush=True)
            continue
        if model is None:
            model = RigidMaskModel(str(args.rigidmask_weights), str(args.midas_weights),
                                   device=args.device, test_resolution=args.resolution)
            model.diagnostics_enabled = False
        baseline_pose = dict(rotation=data['geometry_rot'], translation=data['geometry_trans'],
                             essential=data['geometry_essential'])
        results = {}
        for name, pose in [('saved_visual_pose', baseline_pose), ('vehicle_xy_yaw', alternative)]:
            results[name] = model.infer(data['previous_bgr'], data['current_bgr'],
                                       data['camera_matrix'], replay_pose=pose)
        valid = data['gated_mask'] != 127
        report = dict(capture=str(path.resolve()), metadata=metadata,
                      lidar_origin_vehicle=args.lidar_origin_vehicle,
                      device=args.device, resolution=args.resolution,
                      runtime_thresholds=thresholds,
                      warning='Planar pose sensitivity test, not ground-truth accuracy; no mask morphology applied',
                      saved_probability_mae=float(np.mean(np.abs(results['saved_visual_pose']-data['probability']))),
                      all_valid={k: statistics(v, valid, thresholds) for k, v in results.items()},
                      originally_dynamic={
                          k: statistics(v, data['gated_mask'] == 255, thresholds)
                          for k, v in results.items()})
        np.savez(output/f'{index:02d}_probabilities.npz', **results,
                 valid_mask=valid, previous_bgr=data['previous_bgr'])
        with open(output/f'{index:02d}_summary.json', 'x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
