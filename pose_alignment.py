# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Align supplied VIO to visual map fixes; T_A_B maps coordinates in B to A.

All timestamps are nonnegative integer nanoseconds in one session's device
clock, not wall time. The caller supplies calibrated device poses and corrected
image-capture timestamps, and serializes calls (including SDK callbacks).
No VIO estimation, ground truth, camera conversion, or SDK access happens here.
"""

from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from map_utils import FloatArray, inverse_transform, rigid_transform
from scipy.spatial.transform import Rotation, Slerp

AlignmentState = Literal["tracking", "stale", "lost"]


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _session(session_uid: str) -> str:
    if not isinstance(session_uid, str) or not session_uid.strip():
        raise ValueError("session_uid must be a nonempty string")
    return session_uid



def vio_status_valid(vio: object) -> bool:
    """Regular VIO (FrontendOutput) is usable only when its status is VALID.

    A GOOD pose_quality can accompany FILTER_NOT_INITIALIZED or INVALID status.
    Records without a status field (e.g. high-frequency poses) are accepted.
    """
    status = getattr(vio, "status", None)
    return status is None or str(status).split(".")[-1] == "VALID"

@dataclass(frozen=True)
class VioSample:
    """One supplied VIO observation; bad-quality matrices are never consumed."""

    timestamp_ns: int
    T_odometry_device: FloatArray
    session_uid: str
    quality_good: bool = True


@dataclass(frozen=True)
class MappedPose:
    """Raw propagated device pose, including capture-based visual-fix freshness.

    A stale pose remains numeric but is NOT a fresh localization. age_ns is the
    device-clock difference from the last accepted fix, not callback latency.
    """

    timestamp_ns: int
    T_map_device: FloatArray
    fix_timestamp_ns: int
    session_uid: str
    age_ns: int
    status: Literal["tracking", "stale"]


class PoseAlignment:
    """Bounded, session-safe VIO history and the latest odometry-to-map alignment.

    Only add_vio establishes a session. A new UID clears history/alignment even
    for a bad-quality observation. UIDs must identify unique odometry frames;
    delayed VIO from retired UIDs cannot reactivate an old session.

    Invalid timestamps, session IDs, good-quality SE(3), or configuration raise
    ValueError. Duplicate/out-of-order VIO returns None without changing state.
    Bad-quality VIO returns None, marks state lost, and blocks interpolation
    across that observation; recovery in the same frame can reuse alignment.
    Fixes lacking valid brackets, from other sessions, or not newer than the
    accepted fix return False and do not alter alignment. No fixes are queued.
    """

    def __init__(
        self,
        *,
        buffer_duration_ns: int = 30_000_000_000,
        max_samples: int = 50_000,
        max_interpolation_gap_ns: int = 250_000_000,
        max_fix_age_ns: int = 120_000_000_000,
    ) -> None:
        self._buffer_duration_ns = _integer(buffer_duration_ns, "buffer_duration_ns", 1)
        self._max_samples = _integer(max_samples, "max_samples", 2)
        self._max_interpolation_gap_ns = _integer(
            max_interpolation_gap_ns, "max_interpolation_gap_ns", 1
        )
        self._max_fix_age_ns = _integer(max_fix_age_ns, "max_fix_age_ns")
        if self._max_interpolation_gap_ns > 250_000_000:
            raise ValueError("max_interpolation_gap_ns must be <= 250000000")
        if self._max_fix_age_ns > 120_000_000_000:
            raise ValueError("max_fix_age_ns must be <= 120000000000")
        self._vio: deque[VioSample] = deque()
        self._retired_sessions: set[str] = set()
        self._session_uid: str | None = None
        self._T_map_odometry: FloatArray | None = None
        self._fix_timestamp_ns: int | None = None

    @property
    def state(self) -> AlignmentState:
        """State at the latest accepted VIO timestamp; no wall-clock timeout."""
        if (
            not self._vio
            or not self._vio[-1].quality_good
            or self._fix_timestamp_ns is None
        ):
            return "lost"
        age_ns = self._vio[-1].timestamp_ns - self._fix_timestamp_ns
        return "tracking" if age_ns <= self._max_fix_age_ns else "stale"

    def add_vio(self, sample: VioSample) -> MappedPose | None:
        """Buffer a strictly newer observation and propagate when aligned.

        Inputs are snapshotted; later mutation of the caller's matrix cannot
        change history. A bad-quality matrix is ignored, even if not finite.
        """
        timestamp_ns = _integer(sample.timestamp_ns, "timestamp_ns")
        session_uid = _session(sample.session_uid)
        if not isinstance(sample.quality_good, bool):
            raise ValueError("quality_good must be a bool")
        if session_uid in self._retired_sessions:
            return None
        if session_uid != self._session_uid:
            if self._session_uid is not None:
                self._retired_sessions.add(self._session_uid)
            self._session_uid = session_uid
            self._vio.clear()
            self._T_map_odometry = None
            self._fix_timestamp_ns = None
        pose = (
            rigid_transform(sample.T_odometry_device).copy()
            if sample.quality_good
            else np.eye(4)
        )
        if self._vio and timestamp_ns <= self._vio[-1].timestamp_ns:
            return None
        self._vio.append(
            VioSample(timestamp_ns, pose, session_uid, sample.quality_good)
        )
        cutoff_timestamp_ns = timestamp_ns - self._buffer_duration_ns
        while (
            len(self._vio) > self._max_samples
            or self._vio[0].timestamp_ns < cutoff_timestamp_ns
        ):
            self._vio.popleft()
        if not sample.quality_good or self._T_map_odometry is None:
            return None
        # Alignment and its fix timestamp are installed together by add_fix.
        assert self._fix_timestamp_ns is not None
        age_ns = timestamp_ns - self._fix_timestamp_ns
        return MappedPose(
            timestamp_ns=timestamp_ns,
            T_map_device=rigid_transform(self._T_map_odometry @ pose),
            fix_timestamp_ns=self._fix_timestamp_ns,
            session_uid=session_uid,
            age_ns=age_ns,
            status="tracking" if age_ns <= self._max_fix_age_ns else "stale",
        )

    def add_fix(
        self,
        corrected_timestamp_ns: int,
        T_map_device: FloatArray,
        session_uid: str,
    ) -> bool:
        """Apply a newer visual fix at its image timestamp, never at receipt time.

        Exact good VIO is preferred. Otherwise require two good adjacent samples
        separated by at most max_interpolation_gap_ns; never extrapolate.
        """
        corrected_timestamp_ns = _integer(
            corrected_timestamp_ns, "corrected_timestamp_ns"
        )
        session_uid = _session(session_uid)
        map_pose = rigid_transform(T_map_device)
        if session_uid != self._session_uid or (
            self._fix_timestamp_ns is not None
            and corrected_timestamp_ns <= self._fix_timestamp_ns
        ):
            return False
        odometry_pose = self._odometry_at(corrected_timestamp_ns)
        if odometry_pose is None:
            return False
        alignment = rigid_transform(map_pose @ inverse_transform(odometry_pose))
        self._T_map_odometry = alignment.copy()
        self._fix_timestamp_ns = corrected_timestamp_ns
        return True

    def _odometry_at(self, timestamp_ns: int) -> FloatArray | None:
        insertion = bisect.bisect_left(
            [sample.timestamp_ns for sample in self._vio], timestamp_ns
        )
        if (
            insertion < len(self._vio)
            and self._vio[insertion].timestamp_ns == timestamp_ns
        ):
            sample = self._vio[insertion]
            return sample.T_odometry_device if sample.quality_good else None
        if insertion == 0 or insertion == len(self._vio):
            return None
        left, right = self._vio[insertion - 1], self._vio[insertion]
        gap_duration_ns = right.timestamp_ns - left.timestamp_ns
        if (
            not left.quality_good
            or not right.quality_good
            or gap_duration_ns > self._max_interpolation_gap_ns
        ):
            return None
        # Subtract integer timestamps before float conversion, preserving ns
        # precision even for device timestamps above float64's integer range.
        fraction = (timestamp_ns - left.timestamp_ns) / gap_duration_ns
        pose = np.eye(4)
        pose[:3, 3] = (1 - fraction) * left.T_odometry_device[
            :3, 3
        ] + fraction * right.T_odometry_device[:3, 3]
        rotations = Rotation.from_matrix(
            np.stack([left.T_odometry_device[:3, :3], right.T_odometry_device[:3, :3]])
        )
        pose[:3, :3] = Slerp([0.0, 1.0], rotations)(fraction).as_matrix()
        return rigid_transform(pose)
