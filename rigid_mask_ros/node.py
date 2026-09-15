"""ROS1 node publishing LiDAR-supported mono8 dynamic masks."""

from collections import deque
import json
import io
import os
import tempfile
import threading
import time
import traceback

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from .model import RigidMaskModel, load_camera_matrices
from .vendor.rigidmask.models.submodule import F_ngransac

# mono8 protocol shared with dh_lidar: 0 static, 127 unavailable, 255 dynamic.
UNKNOWN = np.uint8(127)


class RigidMaskNode:
    def __init__(self):
        self.camera_ids = [int(value) for value in rospy.get_param(
            "~camera_ids", [1])]
        self.input_pattern = rospy.get_param(
            "~input_topic_pattern", "/camera/{camera_id}/image_raw/compressed")
        self.output_pattern = rospy.get_param(
            "~output_topic_pattern", "/rigid_mask/camera_{camera_id}/dynamic_mask")
        self.support_pattern = rospy.get_param(
            "~lidar_support_topic_pattern",
            "/rigid_mask/camera_{camera_id}/lidar_support")
        self.use_lidar_support = bool(rospy.get_param(
            "~use_lidar_support", True))
        self.support_max_delta = float(rospy.get_param(
            "~lidar_support_max_delta", 0.001))
        self.support_dilation = max(0, int(rospy.get_param(
            "~lidar_support_dilation", 15)))
        self.debug_pattern = rospy.get_param(
            "~debug_topic_pattern", "/rigid_mask/camera_{camera_id}/debug/compressed")
        self.publish_debug = bool(rospy.get_param("~publish_debug", False))
        self.publish_diagnostics = bool(rospy.get_param("~publish_diagnostics", True))
        # Per-run, bounded replay captures; no environment changes or old-file deletion.
        self.capture_limit = max(0, min(60, int(rospy.get_param("~diagnostic_capture_limit", 30))))
        self.capture_interval = float(rospy.get_param("~diagnostic_capture_interval", 8.0))
        if not np.isfinite(self.capture_interval) or not 1.0 <= self.capture_interval <= 60.0:
            raise ValueError('diagnostic_capture_interval must be 1..60 seconds')
        self.capture_bytes = 0
        self.capture_count = 0
        self.capture_directory = None
        self.capture_last_time = -float('inf')
        self.queue_size = int(rospy.get_param("~queue_size", 1))
        self.frame_skip = max(0, int(rospy.get_param("~frame_skip", 0)))
        self.pair_target_delta = max(0.01, float(rospy.get_param(
            "~pair_target_delta", 0.10)))
        self.pair_min_delta = max(0.0, float(rospy.get_param(
            "~pair_min_delta", 0.075)))
        self.pair_max_delta = max(
            self.pair_min_delta, float(rospy.get_param(
                "~pair_max_delta", 0.20)))
        self.threshold = float(rospy.get_param("~dynamic_threshold", 0.5))
        self.probability_margin = float(rospy.get_param("~probability_margin", 0.10))
        if not (0 <= self.probability_margin < min(self.threshold, 1 - self.threshold)):
            raise ValueError("probability_margin must leave static/dynamic confidence ranges")
        self.min_area = max(0, int(rospy.get_param("~min_dynamic_area", 80)))
        self.close_kernel = max(0, int(rospy.get_param("~mask_close_kernel", 5)))
        if rospy.get_param("~fail_closed_dynamic", False):
            rospy.logwarn("fail_closed_dynamic is ignored: failures are UNKNOWN, not motion")
        self.stationary_camera_mode = bool(rospy.get_param(
            "~stationary_camera_mode", True))
        self.motion_analysis_scale = min(1.0, max(0.2, float(
            rospy.get_param("~motion_analysis_scale", 0.5))))
        self.motion_reference_dt = max(0.01, float(rospy.get_param(
            "~motion_reference_dt", 0.05)))
        self.stationary_enter_flow = max(0.0, float(rospy.get_param(
            "~stationary_enter_flow_px", 0.12)))
        self.stationary_exit_flow = max(
            self.stationary_enter_flow, float(rospy.get_param(
                "~stationary_exit_flow_px", 0.35)))
        self.stationary_difference = max(0, int(rospy.get_param(
            "~stationary_difference_threshold", 8)))
        self.stationary_min_tracks = max(8, int(rospy.get_param(
            "~stationary_min_tracks", 30)))
        self.stationary_motion_dilation = max(0, int(rospy.get_param(
            "~stationary_motion_dilation", 5)))
        self.stationary_min_area = max(self.min_area, int(rospy.get_param(
            "~stationary_min_dynamic_area", 250)))
        self.pose_topic = rospy.get_param("~pose_topic", "/current_pose")
        self.pose_max_delta = max(0.05, float(rospy.get_param(
            "~pose_max_delta", 0.30)))
        self.ego_stationary_speed = max(0.0, float(rospy.get_param(
            "~ego_stationary_speed_mps", 0.15)))
        self.ego_moving_speed = max(self.ego_stationary_speed, float(rospy.get_param(
            "~ego_moving_speed_mps", 0.75)))
        self.stationary_enter_seconds = max(0.0, float(rospy.get_param(
            "~stationary_enter_seconds", 0.30)))
        self.stationary_exit_seconds = max(0.0, float(rospy.get_param(
            "~stationary_exit_seconds", 0.25)))
        pose_estimator = rospy.get_param("~pose_estimator", "opencv")
        if pose_estimator != "opencv":
            raise ValueError("Only the isolated 'opencv' pose estimator is supported")

        calibration_file = rospy.get_param("~calibration_file")
        with open(calibration_file, 'r', encoding='utf-8') as stream:
            self.diagnostic_calibration_yaml = stream.read()
        self.camera_matrices = load_camera_matrices(calibration_file)
        missing_calibration = [value for value in self.camera_ids
                               if value not in self.camera_matrices]
        if missing_calibration:
            raise RuntimeError(
                "Missing calibration for cameras {} in {}".format(
                    missing_calibration, calibration_file))

        self.model = RigidMaskModel(
            rigidmask_weights=rospy.get_param("~rigidmask_weights"),
            midas_weights=rospy.get_param("~midas_weights"),
            device=rospy.get_param("~device", "cuda:0"),
            test_resolution=float(rospy.get_param("~test_resolution", 0.5)),
            pose_ransac_threshold=float(rospy.get_param(
                "~ransac_threshold", 0.0015)))

        self.publishers = {
            camera_id: rospy.Publisher(
                self.output_pattern.format(camera_id=camera_id), Image,
                queue_size=self.queue_size)
            for camera_id in self.camera_ids
        }
        self.debug_publishers = {
            camera_id: rospy.Publisher(
                self.debug_pattern.format(camera_id=camera_id), CompressedImage,
                queue_size=self.queue_size)
            for camera_id in self.camera_ids
        } if self.publish_debug else {}
        self.reference_publishers = {
            camera_id: rospy.Publisher(
                "/rigid_mask/camera_{}/reference_image/compressed".format(camera_id),
                CompressedImage, queue_size=20)
            for camera_id in self.camera_ids
        }
        self.diagnostic_publishers = {
            camera_id: {
                "probability": rospy.Publisher(
                    "/rigid_mask/camera_{}/raw_probability".format(camera_id), Image, queue_size=1),
                "mask": rospy.Publisher(
                    "/rigid_mask/camera_{}/raw_dynamic_mask".format(camera_id), Image, queue_size=1),
                "info": rospy.Publisher(
                    "/rigid_mask/camera_{}/diagnostics".format(camera_id), String, queue_size=10),
            } for camera_id in self.camera_ids
        } if self.publish_diagnostics else {}

        self.condition = threading.Condition()
        self.model.diagnostics_enabled = self.publish_diagnostics
        self.raw_mask_diagnostics = None
        self.frame_pair_diagnostics = None
        rospy.loginfo('RIGID_CONFIG calibration_file=%s K=%s resolution=%s ransac=%s '
                      'stationary_flow_enter=%s exit=%s ego_enter=%s exit=%s '
                      'support_dt=%s threshold=%s margin=%s',
                      calibration_file,
                      {k: v.tolist() for k, v in self.camera_matrices.items()},
                      self.model.test_resolution, self.model.model.pose_ransac_threshold,
                      self.stationary_enter_flow, self.stationary_exit_flow,
                      self.ego_stationary_speed, self.ego_moving_speed,
                      self.support_max_delta, self.threshold, self.probability_margin)
        self.pending = {}
        self.frame_history = {
            camera_id: deque(maxlen=20) for camera_id in self.camera_ids
        }
        self.lidar_support = {
            camera_id: deque(maxlen=80) for camera_id in self.camera_ids
        }
        self.frame_counts = {camera_id: 0 for camera_id in self.camera_ids}
        self.camera_stationary = {
            camera_id: None for camera_id in self.camera_ids
        }
        self.mode_timing = {camera_id: (None, 0.0, 0.0) for camera_id in self.camera_ids}
        self.pose_lock = threading.Lock()
        self.pose_history = deque(maxlen=200)
        self.next_camera_index = 0
        self.stopping = False
        self.subscribers = [
            rospy.Subscriber(
                self.input_pattern.format(camera_id=camera_id), CompressedImage,
                self._image_callback, callback_args=camera_id,
                queue_size=self.queue_size, buff_size=16 * 1024 * 1024)
            for camera_id in self.camera_ids
        ]
        if self.use_lidar_support:
            self.subscribers.extend(
                rospy.Subscriber(
                    self.support_pattern.format(camera_id=camera_id), Image,
                    self._support_callback, callback_args=camera_id,
                    queue_size=10)
                for camera_id in self.camera_ids
            )
        self.subscribers.append(rospy.Subscriber(
            self.pose_topic, PointStamped, self._pose_callback,
            queue_size=50))
        self.worker = threading.Thread(
            target=self._worker_loop, name="rigid-mask-inference", daemon=True)
        self.worker.start()
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo(
            "rigid_mask_ros ready: cameras=%s device=%s threshold=%.3f",
            self.camera_ids, self.model.device, self.threshold)

    def _pose_callback(self, message):
        stamp = message.header.stamp
        point = message.point
        if (stamp == rospy.Time() or
                not np.isfinite([point.x, point.y]).all()):
            rospy.logwarn_throttle(
                1.0, "RigidMask rejected invalid ego pose")
            return
        sample = (stamp.to_sec(), float(point.x), float(point.y),
                  float(point.z) if np.isfinite(point.z) else None)
        with self.pose_lock:
            if self.pose_history and sample[0] <= self.pose_history[-1][0]:
                if sample[0] < self.pose_history[-1][0]:
                    self.pose_history.clear()
                else:
                    return
            self.pose_history.append(sample)

    def _ego_speed_at(self, stamp):
        """Return robust ego speed near an image stamp, or None if unavailable."""
        target = stamp.to_sec()
        with self.pose_lock:
            samples = list(self.pose_history)
        speeds = []
        for before, after in zip(samples, samples[1:]):
            dt = after[0] - before[0]
            midpoint = 0.5 * (before[0] + after[0])
            if (0.005 <= dt <= 0.50 and
                    abs(midpoint - target) <= self.pose_max_delta):
                speeds.append(np.hypot(
                    after[1] - before[1], after[2] - before[2]) / dt)
        if not speeds:
            return None
        return float(np.median(speeds[-9:]))

    def _image_callback(self, message, camera_id):
        # Match traffic.py's camera path: keep the newest CompressedImage in a
        # queue_size=1 callback and decode outside the ROS callback thread.
        if not message.data or message.header.stamp == rospy.Time():
            rospy.logwarn_throttle(
                1.0, "RigidMask camera %d compressed image is empty", camera_id)
            return
        self.frame_counts[camera_id] += 1
        if (self.frame_counts[camera_id] - 1) % (self.frame_skip + 1):
            return
        # Both inference and LiDAR support now see this exact image stream.
        self.reference_publishers[camera_id].publish(message)
        with self.condition:
            history = self.frame_history[camera_id]
            if (history and message.header.stamp <= history[-1].header.stamp):
                history.clear()
            candidates = []
            for sample in history:
                delta = (message.header.stamp - sample.header.stamp).to_sec()
                if self.pair_min_delta <= delta <= self.pair_max_delta:
                    candidates.append((
                        abs(delta - self.pair_target_delta), sample))
            previous = min(candidates, key=lambda item: item[0])[1] \
                if candidates else None
            history.append(message)
            self.pending[camera_id] = (previous, message)
            self.condition.notify()

    def _support_callback(self, message, camera_id):
        if (message.header.stamp == rospy.Time() or
                message.encoding not in ("mono8", "8UC1") or
                message.width == 0 or message.height == 0 or
                message.step < message.width or
                len(message.data) < message.step * message.height):
            rospy.logwarn_throttle(
                1.0, "RigidMask camera %d rejected invalid LiDAR support",
                camera_id)
            return
        with self.condition:
            self.lidar_support[camera_id].append(message)
            self.condition.notify_all()

    def _closest_support(self, camera_id, stamp):
        if not self.use_lidar_support:
            return None
        with self.condition:
            samples = list(self.lidar_support[camera_id])
        if not samples:
            return None
        best = min(samples, key=lambda item: abs(
            (item.header.stamp - stamp).to_sec()))
        if abs((best.header.stamp - stamp).to_sec()) > self.support_max_delta:
            return None
        raw = np.frombuffer(best.data, dtype=np.uint8).reshape(
            best.height, best.step)[:, :best.width]
        return np.ascontiguousarray(raw)

    def _apply_lidar_support(self, camera_id, header, mask):
        if not self.use_lidar_support:
            return mask
        support = self._closest_support(camera_id, header.stamp)
        deadline = time.monotonic() + 0.10
        # The stationary path can finish before the pose/LiDAR callbacks have
        # projected this camera frame. Wait briefly without blocking callbacks.
        while support is None and time.monotonic() < deadline and not self.stopping:
            with self.condition:
                self.condition.wait(timeout=min(0.02, max(0.0, deadline - time.monotonic())))
            support = self._closest_support(camera_id, header.stamp)
        if support is None:
            rospy.logwarn_throttle(
                1.0,
                "RigidMask camera %d has no time-matched LiDAR support; "
                "mask publication skipped",
                camera_id)
            return None
        if support.shape != mask.shape:
            support = cv2.resize(
                support, (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_NEAREST)
        if self.support_dilation >= 3:
            size = self.support_dilation | 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (size, size))
            support = cv2.dilate(support, kernel)
        gated = mask.copy()
        gated[support == 0] = UNKNOWN
        return gated

    def _take_next(self):
        for offset in range(len(self.camera_ids)):
            index = (self.next_camera_index + offset) % len(self.camera_ids)
            camera_id = self.camera_ids[index]
            if camera_id in self.pending:
                self.next_camera_index = (index + 1) % len(self.camera_ids)
                previous, current = self.pending.pop(camera_id)
                return camera_id, previous, current
        return None

    @staticmethod
    def _decode(message):
        encoded = np.frombuffer(message.data, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def _camera_motion(self, camera_id, previous, current, delta_time,
                       ego_speed, stamp_seconds=None):
        """Detect a degenerate stationary-camera pair before RigidMask pose."""
        scale = self.motion_analysis_scale
        size = (max(1, int(current.shape[1] * scale)),
                max(1, int(current.shape[0] * scale)))
        previous_gray = cv2.resize(
            cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), size,
            interpolation=cv2.INTER_AREA)
        current_gray = cv2.resize(
            cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), size,
            interpolation=cv2.INTER_AREA)

        corners = cv2.goodFeaturesToTrack(
            previous_gray, maxCorners=600, qualityLevel=0.01,
            minDistance=8, blockSize=7)
        affine = np.asarray([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0]], dtype=np.float32)
        tracked_count = 0
        median_flow = float("inf")
        if corners is not None:
            tracked, status, error = cv2.calcOpticalFlowPyrLK(
                previous_gray, current_gray, corners, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS |
                          cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            if tracked is not None and status is not None:
                valid = status.reshape(-1).astype(bool)
                if error is not None:
                    valid &= np.isfinite(error.reshape(-1))
                source = corners.reshape(-1, 2)[valid]
                destination = tracked.reshape(-1, 2)[valid]
                finite = (np.isfinite(source).all(axis=1) &
                          np.isfinite(destination).all(axis=1))
                source = source[finite]
                destination = destination[finite]
                tracked_count = len(source)
                if tracked_count:
                    time_scale = self.motion_reference_dt / max(
                        1e-3, min(delta_time, 0.5))
                    displacement = np.linalg.norm(
                        destination - source, axis=1)
                    median_flow = float(np.median(displacement))
                    median_flow *= time_scale / scale
                if tracked_count >= self.stationary_min_tracks:
                    estimate, _ = cv2.estimateAffinePartial2D(
                        source, destination, method=cv2.RANSAC,
                        ransacReprojThreshold=1.5, maxIters=1000,
                        confidence=0.99, refineIters=10)
                    if estimate is not None and np.isfinite(estimate).all():
                        affine = estimate.astype(np.float32)

        if tracked_count < self.stationary_min_tracks:
            difference = cv2.absdiff(previous_gray, current_gray)
            changed_fraction = float(np.mean(
                difference >= self.stationary_difference))
            if changed_fraction <= 0.02:
                median_flow = 0.0

        # Small image flow alone is not enough: while creeping past a nearby
        # pole, distant background features barely move and previously put the
        # node in stationary mode. That made newly revealed static surfaces
        # look like independently moving objects. Ego pose is used only to
        # choose the correct RigidMask inference path, never as object-motion
        # evidence.
        stationary = self._select_camera_mode(
            camera_id, median_flow, ego_speed,
            time.monotonic() if stamp_seconds is None else stamp_seconds)
        return (stationary, median_flow, tracked_count, affine,
                previous_gray, current_gray)

    def _select_camera_mode(self, camera_id, flow, ego_speed, stamp):
        last, entering, leaving = self.mode_timing[camera_id]
        dt = 0.0 if last is None else max(0.0, min(0.5, stamp - last))
        state = bool(self.camera_stationary[camera_id])
        if last is not None and (stamp <= last or stamp - last > 0.5):
            entering = leaving = 0.0
            state = False
            dt = 0.0
        strong_motion = (not np.isfinite(flow) or flow > self.stationary_exit_flow or
                         (ego_speed is not None and ego_speed >= self.ego_moving_speed))
        stopped = (flow <= self.stationary_enter_flow and
                   (ego_speed is None or ego_speed <= self.ego_stationary_speed))
        if strong_motion:
            state = False
            entering = leaving = 0.0
        elif stopped:
            entering += dt
            leaving = 0.0
            if entering >= self.stationary_enter_seconds:
                state = True
        else:
            entering = 0.0
            leaving += dt
            if leaving >= self.stationary_exit_seconds:
                state = False
        self.mode_timing[camera_id] = (stamp, entering, leaving)
        self.camera_stationary[camera_id] = state
        return state

    def _stationary_motion_mask(self, previous_gray, current_gray, affine,
                                delta_time, output_shape):
        """Segment local motion after compensating small global vibration."""
        height, width = previous_gray.shape
        aligned_previous = cv2.warpAffine(
            previous_gray, affine, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        difference = cv2.absdiff(
            cv2.GaussianBlur(aligned_previous, (5, 5), 0),
            cv2.GaussianBlur(current_gray, (5, 5), 0))

        # Dense Farneback flow took about one second on the vehicle computer
        # and compared duplicate simulator frames. The 0.1 s input baseline
        # plus globally aligned photometric residual is both faster and more
        # sensitive to a crossing vehicle.
        mask = (difference >= self.stationary_difference).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        analysis_min_area = max(1, int(np.ceil(
            self.stationary_min_area * self.motion_analysis_scale ** 2)))
        mask = self._remove_small_components(mask, analysis_min_area)
        if self.stationary_motion_dilation >= 3:
            size = self.stationary_motion_dilation | 1
            mask = cv2.dilate(
                mask, cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (size, size)))
        mask = cv2.resize(
            mask, (output_shape[1], output_shape[0]),
            interpolation=cv2.INTER_NEAREST)
        mask = self._filter_binary_mask(mask, self.stationary_min_area)
        # Pixels exposed by the warp have no previous observation. Replicated
        # image borders must not become static/dynamic evidence.
        valid = cv2.warpAffine(
            np.full(previous_gray.shape, 255, np.uint8), affine, (width, height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.erode(valid, np.ones((5, 5), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.resize(valid, (output_shape[1], output_shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        mask[valid == 0] = UNKNOWN
        return mask

    def _worker_loop(self):
        while not rospy.is_shutdown():
            with self.condition:
                self.condition.wait_for(
                    lambda: self.stopping or bool(self.pending), timeout=0.5)
                if self.stopping:
                    return
                work = self._take_next()
            if work is None:
                continue
            camera_id, previous_message, current_message = work
            started = time.monotonic()
            self.model.last_diagnostics = {}
            F_ngransac.last_diagnostics = {}
            self.raw_mask_diagnostics = None
            self.frame_pair_diagnostics = None
            # Re-send the selected pair as well: callbacks may have dropped
            # intermediate frames under load, but support must use this pair.
            if previous_message is not None:
                self.reference_publishers[camera_id].publish(previous_message)
            self.reference_publishers[camera_id].publish(current_message)
            current = self._decode(current_message)
            if current is None:
                rospy.logerr_throttle(
                    1.0, "RigidMask camera %d image decode failed", camera_id)
                continue
            if previous_message is None:
                self._publish_unknown(camera_id, current_message.header, current.shape[:2])
                self._diagnostics(camera_id, current_message.header, None, current_message,
                                  started, "no_frame_pair", None, None)
                continue
            fallback_context = None
            output_header = previous_message.header
            reference_image = current
            try:
                previous = self._decode(previous_message)
                if previous is None:
                    raise ValueError("previous compressed image decode failed")
                reference_image = previous
                if previous.shape != current.shape:
                    raise ValueError("camera resolution changed between frame pair")
                identical = np.array_equal(previous, current)
                if self.publish_diagnostics:
                    self.frame_pair_diagnostics = dict(
                        identical=bool(identical),
                        mean_abs_difference_sampled=float(np.mean(cv2.absdiff(
                            previous[::8, ::8], current[::8, ::8]))))
                delta_time = max(
                    1e-3, (current_message.header.stamp -
                           previous_message.header.stamp).to_sec())
                ego_speed = self._ego_speed_at(current_message.header.stamp)
                stationary = False
                median_flow = float("inf")
                tracked_count = 0
                if self.stationary_camera_mode:
                    (stationary, median_flow, tracked_count, affine,
                     previous_gray, current_gray) = self._camera_motion(
                         camera_id, previous, current, delta_time, ego_speed,
                         current_message.header.stamp.to_sec())
                    fallback_context = (
                        previous_gray, current_gray, affine, delta_time,
                        ego_speed, median_flow)
                else:
                    fallback_context = None
                if identical and not stationary:
                    self._publish_unknown(camera_id, output_header, previous.shape[:2])
                    self._diagnostics(camera_id, output_header, previous_message, current_message,
                                      started, "duplicate_or_mode_transition", ego_speed, median_flow)
                    continue
                if stationary:
                    output_header = current_message.header
                    reference_image = current
                    mask = self._stationary_motion_mask(
                        previous_gray, current_gray, affine, delta_time,
                        current.shape[:2])
                    probability = mask.astype(np.float32) / 255.0
                    probability[mask == UNKNOWN] = np.nan
                    rospy.loginfo_throttle(
                        1.0,
                        "RigidMask camera %d stationary mode: "
                        "flow=%.3f px/%.0fms ego=%.2f m/s tracks=%d "
                        "local_pixels=%d",
                        camera_id, median_flow,
                        self.motion_reference_dt * 1000.0,
                        -1.0 if ego_speed is None else ego_speed,
                        tracked_count,
                        int(np.count_nonzero(mask == 255)))
                else:
                    capture = (self.publish_diagnostics and self.capture_count < self.capture_limit
                               and time.monotonic() - self.capture_last_time >= self.capture_interval)
                    probability = self.model.infer(
                        previous, current, self.camera_matrices[camera_id], capture_spatial=capture)
                    mask = self._postprocess(probability)
                    rospy.loginfo_throttle(
                        1.0,
                        "RigidMask camera %d moving mode: "
                        "flow=%.3f px/%.0fms ego=%.2f m/s tracks=%d",
                        camera_id, median_flow,
                        self.motion_reference_dt * 1000.0,
                        -1.0 if ego_speed is None else ego_speed,
                        tracked_count)
                # Neural results use previous-frame coordinates; stationary
                # image differences use current-frame coordinates. Never OR
                # masks from different frames without registering them.
                self._publish_raw(camera_id, output_header, probability, mask)
                mask = self._apply_lidar_support(
                    camera_id, output_header, mask)
                if mask is None:
                    self._publish_unknown(camera_id, output_header, reference_image.shape[:2])
                    self._diagnostics(camera_id, output_header, previous_message, current_message,
                                      started, "missing_support", ego_speed, median_flow)
                    continue
                self._publish_mask(camera_id, output_header, mask)
                if not stationary and capture:
                    self._save_replay_capture(camera_id, previous_message, current_message,
                                              previous, current, probability, mask)
                self._diagnostics(camera_id, output_header, previous_message, current_message,
                                  started, "stationary" if stationary else "moving", ego_speed,
                                  median_flow, mask)
                rospy.loginfo_throttle(
                    1.0, "RigidMask output: ref=%.6f pair_dt=%.3fs processing=%.3fs "
                    "age=%.3fs valid=%.3f pose=%s",
                    output_header.stamp.to_sec(), delta_time, time.monotonic() - started,
                    (rospy.Time.now() - output_header.stamp).to_sec(),
                    float(np.mean(mask != UNKNOWN)),
                    "stationary" if stationary else getattr(F_ngransac, "last_diagnostics", {}))
                if self.publish_debug:
                    self._publish_debug(
                        camera_id, output_header, reference_image,
                        probability, mask)
            except Exception as error:
                self._diagnostics(camera_id, output_header, previous_message, current_message,
                                  started, "error: " + str(error), None, None)
                if str(error).startswith("OpenCV pose"):
                    rospy.logwarn_throttle(
                        2.0, "RigidMask camera %d pose unavailable: %s; "
                        "checking whether image-motion fallback is safe",
                        camera_id, error)
                else:
                    rospy.logerr_throttle(
                        2.0, "RigidMask camera %d inference failed: %s\n%s",
                        camera_id, error, traceback.format_exc())
                if fallback_context is not None:
                    (previous_gray, current_gray, affine, delta_time,
                     ego_speed, median_flow) = fallback_context
                    safe_stationary_fallback = bool(self.camera_stationary[camera_id]) and (
                        ego_speed <= self.ego_stationary_speed and
                        median_flow <= self.stationary_exit_flow
                        if ego_speed is not None else
                        median_flow <= self.stationary_enter_flow)
                    if safe_stationary_fallback:
                        mask = self._stationary_motion_mask(
                            previous_gray, current_gray, affine, delta_time,
                            current.shape[:2])
                        output_header = current_message.header
                        mask = self._apply_lidar_support(
                            camera_id, output_header, mask)
                        if mask is not None:
                            self._publish_mask(
                                camera_id, output_header, mask)
                            self._diagnostics(camera_id, output_header, previous_message,
                                              current_message, started, "stationary_fallback",
                                              ego_speed, median_flow, mask)
                            continue
                    else:
                        rospy.logwarn_throttle(
                            2.0, "RigidMask camera %d skipped unsafe "
                            "image-difference fallback while ego is moving",
                            camera_id)
                self._publish_unknown(camera_id, output_header, current.shape[:2])

    @staticmethod
    def _region_diagnostics(arrays, mask):
        """CNN-labelled regions, not ground truth; preserve each map's native grid."""
        names = ['vcn_flow_x_network_px', 'vcn_flow_y_network_px',
                 'depth_ratio_Z0_over_Z1', 'flow_oor_logits',
                 'depth_change_uncertainty', 'midas_inverse_depth_relative',
                 'depth_contrast_log_ratio', 'foreground_contributions',
                 'foreground_residual']
        result = {'regions_are_predictions_not_ground_truth': True, 'maps': {}}
        for name in names:
            if name not in arrays:
                continue
            value = np.asarray(arrays[name])
            h, w = value.shape[-2:]
            channels = value.reshape(-1, h, w)
            resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            regions = {}
            for label, selection in [('dynamic', resized_mask == 255),
                                     ('static', resized_mask == 0)]:
                summaries = []
                for channel in channels:
                    values = channel[selection]
                    # Bound CPU reductions; report sample size explicitly.
                    values = values[::max(1, (values.size + 2047)//2048)]
                    finite = values[np.isfinite(values)]
                    summaries.append(dict(sampled=int(values.size), finite=int(finite.size),
                                          p05_p50_p95=np.percentile(finite, [5,50,95]).tolist()
                                          if finite.size else None))
                regions[label] = summaries
            result['maps'][name] = regions
        return result

    def _save_replay_capture(self, camera_id, previous_message, current_message,
                             previous, current, probability, mask):
        """Diagnostic failures must not suppress or change a normal mask."""
        self.capture_last_time = time.monotonic()
        try:
            arrays = dict(self.model.model.spatial_diagnostics)
            arrays.update(previous_bgr=previous, current_bgr=current,
                          probability=probability, gated_mask=mask,
                          calibration_yaml=np.array(self.diagnostic_calibration_yaml),
                          camera_matrix=self.camera_matrices[camera_id])
            contribution_names = list(
                self.model.model.foreground_contribution_channel_names)
            fgnet_input_names = list(self.model.model.fgnet_input_channel_names)
            metadata = dict(schema_version=2, camera_id=camera_id,
                            previous_stamp=previous_message.header.stamp.to_sec(),
                            current_stamp=current_message.header.stamp.to_sec(),
                            ego_pair=self._diagnostic_ego_pair(previous_message, current_message),
                            pose=getattr(F_ngransac, 'last_diagnostics', {}),
                            contribution_order=contribution_names,
                            foreground_contribution_channel_names=contribution_names,
                            fgnet_input_channel_names=fgnet_input_names,
                            fgnet_input_channel_groups=
                                self.model.model.fgnet_input_channel_groups,
                            foreground_thresholds=dict(
                                base_threshold=self.threshold,
                                probability_margin=self.probability_margin,
                                dynamic_probability_threshold=
                                    self.threshold + self.probability_margin,
                                static_probability_threshold=
                                    self.threshold - self.probability_margin,
                                source='RigidMaskNode._postprocess'),
                            foreground_replay=dict(
                                final_logit_key='foreground_logits',
                                final_probability_key='probability',
                                native_formula='sum(foreground_contributions, channel) + foreground_residual',
                                network_resize='torch.nn.functional.interpolate(mode=bilinear, align_corners=False)',
                                activation='sigmoid',
                                full_frame_resize='cv2.resize(interpolation=INTER_LINEAR)'),
                            tau_conventions=dict(
                                raw_key='optical_expansion_tau_Z1_over_Z0',
                                raw_convention='Z1/Z0',
                                converted_key='depth_ratio_Z0_over_Z1',
                                converted_convention='Z0/Z1'),
                            note='Raw full frames and spatial maps; alternative pose is offline only')
            metadata['tensor_metadata'] = {
                name: {'shape': list(np.asarray(value).shape),
                       'dtype': str(np.asarray(value).dtype)}
                for name, value in arrays.items()
                if name in ('fgnet_input_tensor', 'fgnet_raw_output',
                            'foreground_inputs', 'foreground_contributions',
                            'foreground_residual',
                            'foreground_logits', 'probability',
                            'optical_expansion_tau_Z1_over_Z0',
                            'depth_ratio_Z0_over_Z1')
            }
            metadata['fgnet_input_capture'] = dict(
                key='fgnet_input_tensor', layout='BCHW', batch_axis=0,
                channel_axis=1, spatial_axes=[2, 3],
                channel_names=fgnet_input_names,
                channel_groups=self.model.model.fgnet_input_channel_groups,
                tensor=metadata['tensor_metadata'].get('fgnet_input_tensor'),
                capture_semantics='detached CPU copy; inference tensor was not modified')
            metadata['fgnet_reinference_capture'] = dict(
                module='VCN.fgnet (bfmodule_feat(160, 7))',
                input_key='fgnet_input_tensor', raw_output_key='fgnet_raw_output',
                foreground_hypotheses_key='foreground_inputs',
                output_semantics=dict(
                    weight_channels='raw_output[:, :6] / 20',
                    residual_channel='raw_output[:, 6:7] / 200',
                    final_native_logit='sum(weights * foreground_hypotheses, channel) + residual'))
            metadata['foreground_contribution_capture'] = dict(
                key='foreground_contributions', layout='BCHW', batch_axis=0,
                channel_axis=1, spatial_axes=[2, 3],
                channel_names=contribution_names,
                tensor=metadata['tensor_metadata'].get(
                    'foreground_contributions'))
            metadata['region_diagnostics'] = self._region_diagnostics(arrays, mask)
            metadata['capture_interval_seconds'] = self.capture_interval
            rospy.loginfo('RIGID_REGION_DIAG camera=%d ref=%.9f data=%s', camera_id,
                          metadata['previous_stamp'],
                          json.dumps(metadata['region_diagnostics'], allow_nan=False))
            arrays['metadata_json'] = np.array(json.dumps(metadata, allow_nan=False))
            buffer = io.BytesIO()
            np.savez(buffer, **arrays)
            size = buffer.tell()
            if self.capture_bytes + size > 256 * 1024 * 1024:
                self.capture_limit = self.capture_count
                rospy.logwarn('RIGID_CAPTURE stopped at 256 MiB session budget')
                return
            if self.capture_directory is None:
                self.capture_directory = tempfile.mkdtemp(prefix='rigid_mask_replay_')
            name = 'camera_{}_{}_{}.npz'.format(camera_id,
                previous_message.header.stamp.to_nsec(), self.capture_count)
            path = os.path.join(self.capture_directory, name)
            with open(path, 'xb') as stream:
                stream.write(buffer.getbuffer())
            self.capture_count += 1
            self.capture_bytes += size
            rospy.loginfo('RIGID_CAPTURE path=%s count=%d/%d bytes=%d save_seconds=%.3f',
                          path, self.capture_count, self.capture_limit, self.capture_bytes,
                          time.monotonic() - self.capture_last_time)
        except Exception as error:
            self.capture_limit = self.capture_count
            rospy.logwarn('RIGID_CAPTURE disabled after diagnostic-only failure: %s', error)
        finally:
            self.model.model.spatial_diagnostics = {}

    def _postprocess(self, probability):
        finite = np.isfinite(probability) & (probability >= 0) & (probability <= 1)
        dynamic = finite & (probability >= self.threshold + self.probability_margin)
        filtered = self._filter_binary_mask(dynamic.astype(np.uint8) * 255)
        mask = np.full(probability.shape, UNKNOWN, dtype=np.uint8)
        mask[finite & (probability <= self.threshold - self.probability_margin)] = 0
        mask[dynamic & (filtered != 0)] = 255
        return mask

    def _publish_unknown(self, camera_id, header, shape):
        self._publish_mask(camera_id, header, np.full(shape, UNKNOWN, dtype=np.uint8))

    def _publish_raw(self, camera_id, header, probability, mask):
        publishers = self.diagnostic_publishers.get(camera_id)
        if not publishers:
            return
        self.raw_mask_diagnostics = dict(
            finite_probability_fraction=float(np.mean(np.isfinite(probability))),
            dynamic_fraction=float(np.mean(mask == 255)),
            static_fraction=float(np.mean(mask == 0)),
            unknown_fraction=float(np.mean(mask == UNKNOWN)))
        for name, array, encoding in [("probability", probability.astype(np.float32), "32FC1"),
                                       ("mask", mask, "mono8")]:
            if publishers[name].get_num_connections() == 0:
                continue
            message = Image()
            message.header = header
            message.height, message.width = array.shape
            message.encoding = encoding
            message.is_bigendian = False
            message.step = message.width * array.dtype.itemsize
            message.data = np.ascontiguousarray(array).tobytes()
            publishers[name].publish(message)

    def _diagnostic_ego_pair(self, previous, current):
        """Diagnostic only: bracketed XY/yaw, no change to mode or inference."""
        if previous is None:
            return {'reason': 'no_frame_pair'}
        with self.pose_lock:
            samples = list(self.pose_history)
        def interpolate(stamp):
            for before, after in zip(samples, samples[1:]):
                if before[0] <= stamp <= after[0]:
                    dt = after[0] - before[0]
                    if dt <= 0 or dt > self.pose_max_delta:
                        return None
                    alpha = (stamp - before[0]) / dt
                    yaw = None
                    if before[3] is not None and after[3] is not None:
                        delta = np.arctan2(np.sin(after[3]-before[3]), np.cos(after[3]-before[3]))
                        yaw = float(before[3] + alpha * delta)
                    return [before[1] + alpha*(after[1]-before[1]),
                            before[2] + alpha*(after[2]-before[2]), yaw]
            return None
        first = interpolate(previous.header.stamp.to_sec())
        last = interpolate(current.header.stamp.to_sec())
        if first is None or last is None:
            return {'reason': 'pose_not_bracketed_or_gap'}
        delta_yaw = None
        if first[2] is not None and last[2] is not None:
            delta_yaw = float(np.arctan2(np.sin(last[2]-first[2]), np.cos(last[2]-first[2])))
        return dict(reason='valid', previous_xy_yaw=first, current_xy_yaw=last,
                    displacement_m=float(np.hypot(last[0]-first[0], last[1]-first[1])),
                    delta_yaw_rad=delta_yaw)

    def _diagnostics(self, camera_id, header, previous, current, started, mode, ego, flow, mask=None):
        publishers = self.diagnostic_publishers.get(camera_id)
        if not publishers:
            return
        finite_value = lambda value: float(value) if value is not None and np.isfinite(value) else None
        payload = dict(schema_version=1, camera_id=camera_id,
                       image_flow_source='LK_mode_selector',
                       image_flow_reference_seconds=self.motion_reference_dt,
                       reference_stamp=header.stamp.to_sec(),
                       previous_stamp=previous.header.stamp.to_sec() if previous else None,
                       current_stamp=current.header.stamp.to_sec(), mode=mode,
                       processing_seconds=time.monotonic() - started,
                       age_seconds=(rospy.Time.now() - header.stamp).to_sec(),
                       ego_speed=finite_value(ego), image_flow=finite_value(flow),
                       valid_fraction=float(np.mean(mask != UNKNOWN)) if mask is not None else None,
                       pose=getattr(F_ngransac, "last_diagnostics", {}) if mode == "moving" else None)
        payload.update(pair=self.frame_pair_diagnostics, raw_mask=self.raw_mask_diagnostics,
                       ego_pair=self._diagnostic_ego_pair(previous, current),
                       network=self.model.last_diagnostics,
                       stationary_selected=bool(self.camera_stationary[camera_id]),
                       mode_timing=self.mode_timing[camera_id],
                       gated_dynamic_fraction=float(np.mean(mask == 255)) if mask is not None else None,
                       gated_static_fraction=float(np.mean(mask == 0)) if mask is not None else None)
        if self.model.last_diagnostics:
            payload['pose'] = getattr(F_ngransac, 'last_diagnostics', {})
        def json_safe(value):
            if isinstance(value, dict):
                return {k: json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_safe(v) for v in value]
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                return None
            return value
        encoded = json.dumps(json_safe(payload), allow_nan=False)
        publishers["info"].publish(String(data=encoded))
        # Persist summaries in rosout, not only in a topic that nobody records.
        # Stage snapshots already run at <=1Hz; never throttle those away.
        if self.model.last_diagnostics:
            rospy.loginfo('RIGID_DIAG %s', encoded)
        else:
            rospy.loginfo_throttle(1.0, 'RIGID_DIAG %s', encoded)

    @staticmethod
    def _remove_small_components(mask, min_area):
        if min_area <= 1:
            return mask
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        filtered = np.zeros_like(mask)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == label] = 255
        return filtered

    def _filter_binary_mask(self, mask, min_area=None):
        if self.close_kernel >= 3:
            size = self.close_kernel | 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return self._remove_small_components(
            mask, self.min_area if min_area is None else min_area)

    def _publish_mask(self, camera_id, header, mask):
        message = Image()
        message.header = header
        message.height, message.width = mask.shape
        message.encoding = "mono8"
        message.is_bigendian = False
        message.step = message.width
        message.data = np.ascontiguousarray(mask).tobytes()
        self.publishers[camera_id].publish(message)

    def _publish_debug(self, camera_id, header, image, probability, mask):
        heat = cv2.applyColorMap(
            np.clip(probability * 255.0, 0, 255).astype(np.uint8),
            cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(image, 0.55, heat, 0.45, 0.0)
        overlay[mask == 255] = (
            0.35 * overlay[mask == 255] + 0.65 * np.asarray([0, 0, 255])
        ).astype(np.uint8)
        overlay[mask == UNKNOWN] = image[mask == UNKNOWN]
        ok, encoded = cv2.imencode(".jpg", overlay)
        if ok:
            message = CompressedImage()
            message.header = header
            message.format = "jpeg"
            message.data = encoded.tobytes()
            self.debug_publishers[camera_id].publish(message)

    def shutdown(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)


def main():
    rospy.init_node("rigid_mask")
    try:
        RigidMaskNode()
        rospy.spin()
    except Exception as error:
        rospy.logfatal("rigid_mask_ros startup failed: %s", error)
        raise
