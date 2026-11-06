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

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from pose_alignment import MappedPose, PoseAlignment, VioSample


@pytest.fixture
def make_vio(make_view) -> Callable[..., VioSample]:
    def make(
        timestamp_ns: int,
        xyz=(0.0, 0.0, 0.0),
        yaw_deg: float = 0.0,
        session_uid: str = "session-A",
        quality_good: bool = True,
    ) -> VioSample:
        return VioSample(
            timestamp_ns,
            make_view(xyz=xyz, yaw_deg=yaw_deg).T_world_camera,
            session_uid,
            quality_good,
        )

    return make


def test_exact_fix_frame_direction_and_high_frequency_propagation(make_vio, make_view):
    alignment = PoseAlignment()
    assert alignment.state == "lost"
    assert not alignment.add_fix(100, np.eye(4), "session-A")
    first = make_vio(100, xyz=(1, 2, 3), yaw_deg=30)
    assert alignment.add_vio(first) is None
    T_map_odometry = make_view(xyz=(10, -20, 30), yaw_deg=90).T_world_camera
    assert alignment.add_fix(100, T_map_odometry @ first.T_odometry_device, "session-A")
    assert alignment.state == "tracking"
    current = make_vio(200, xyz=(3, 4, 5), yaw_deg=60)
    mapped = alignment.add_vio(current)
    assert isinstance(mapped, MappedPose)
    np.testing.assert_allclose(
        mapped.T_map_device, T_map_odometry @ current.T_odometry_device, atol=1e-12
    )
    np.testing.assert_allclose(mapped.T_map_device[:3, 3], [15, -16, 27], atol=1e-12)
    assert mapped.timestamp_ns == 200
    assert mapped.fix_timestamp_ns == 100
    assert mapped.age_ns == 100
    assert mapped.session_uid == "session-A"
    assert mapped.status == "tracking"


def test_capture_interpolation_translation_slerp_and_latency(make_vio, make_view):
    alignment = PoseAlignment()
    left = make_vio(0)
    right = make_vio(250_000_000, xyz=(2, 4, 6), yaw_deg=180)
    alignment.add_vio(left)
    alignment.add_vio(right)
    capture_pose = make_view(xyz=(1, 2, 3), yaw_deg=90).T_world_camera
    T_map_odometry = make_view(xyz=(7, 8, 9), yaw_deg=-30).T_world_camera
    assert alignment.add_fix(125_000_000, T_map_odometry @ capture_pose, "session-A")
    current = make_vio(300_000_000, xyz=(3, 5, 7), yaw_deg=160)
    mapped = alignment.add_vio(current)
    assert mapped is not None
    np.testing.assert_allclose(
        mapped.T_map_device, T_map_odometry @ current.T_odometry_device, atol=1e-12
    )
    assert mapped.fix_timestamp_ns == 125_000_000
    assert mapped.age_ns == 175_000_000


@pytest.mark.parametrize("base_timestamp_ns", [0, 2**60])
@pytest.mark.parametrize("offset_ns,expected_yaw_deg", [(1, 175), (2, 180), (3, 185)])
def test_slerp_shortest_arc_and_integer_nanosecond_precision(
    make_vio, make_view, base_timestamp_ns, offset_ns, expected_yaw_deg
):
    alignment = PoseAlignment()
    alignment.add_vio(make_vio(base_timestamp_ns, yaw_deg=170))
    right = make_vio(base_timestamp_ns + 4, xyz=(4, 8, 12), yaw_deg=-170)
    alignment.add_vio(right)
    capture_pose = make_view(
        xyz=(offset_ns, 2 * offset_ns, 3 * offset_ns), yaw_deg=expected_yaw_deg
    ).T_world_camera
    assert alignment.add_fix(base_timestamp_ns + offset_ns, capture_pose, "session-A")
    mapped = alignment.add_vio(
        make_vio(base_timestamp_ns + 5, xyz=(4, 8, 12), yaw_deg=-170)
    )
    assert mapped is not None
    np.testing.assert_allclose(mapped.T_map_device, right.T_odometry_device, atol=1e-12)
    assert mapped.age_ns == 5 - offset_ns


@pytest.mark.parametrize("fix_timestamp_ns", [0, 99, 101, 200])
def test_single_sample_never_extrapolates(make_vio, fix_timestamp_ns):
    alignment = PoseAlignment()
    alignment.add_vio(make_vio(100))
    assert not alignment.add_fix(fix_timestamp_ns, np.eye(4), "session-A")
    assert alignment.state == "lost"


def test_future_fix_is_not_queued_but_can_be_retried_with_brackets(make_vio):
    alignment = PoseAlignment()
    alignment.add_vio(make_vio(0))
    assert not alignment.add_fix(10, np.eye(4), "session-A")
    assert alignment.add_vio(make_vio(20)) is None
    assert alignment.state == "lost"
    assert alignment.add_fix(10, np.eye(4), "session-A")
    assert alignment.add_vio(make_vio(21)) is not None


def test_interpolation_gap_limit_is_inclusive_but_exact_has_no_gap_limit(make_vio):
    alignment = PoseAlignment()
    alignment.add_vio(make_vio(0))
    alignment.add_vio(make_vio(250_000_001))
    assert not alignment.add_fix(125_000_000, np.eye(4), "session-A")
    assert alignment.add_fix(250_000_001, np.eye(4), "session-A")
    smaller_gap = PoseAlignment(max_interpolation_gap_ns=4)
    smaller_gap.add_vio(make_vio(0))
    smaller_gap.add_vio(make_vio(5))
    assert not smaller_gap.add_fix(2, np.eye(4), "session-A")


@pytest.mark.parametrize("fix_timestamp_ns", [1, 2, 3])
def test_bad_quality_exact_or_bracket_is_never_consumed(make_vio, fix_timestamp_ns):
    alignment = PoseAlignment()
    alignment.add_vio(make_vio(0))
    bad = VioSample(2, np.full((4, 4), np.nan), "session-A", False)
    assert alignment.add_vio(bad) is None
    assert alignment.state == "lost"
    alignment.add_vio(make_vio(4))
    assert not alignment.add_fix(fix_timestamp_ns, np.eye(4), "session-A")
    assert alignment.add_fix(4, np.eye(4), "session-A")


def test_quality_loss_is_lost_and_same_frame_recovery_reuses_alignment(make_vio):
    alignment = PoseAlignment()
    alignment.add_vio(make_vio(0))
    map_pose = np.eye(4)
    map_pose[:3, 3] = [10, 20, 30]
    assert alignment.add_fix(0, map_pose, "session-A")
    assert alignment.add_vio(make_vio(1, quality_good=False)) is None
    assert alignment.state == "lost"
    assert not alignment.add_fix(1, map_pose, "session-A")
    mapped = alignment.add_vio(make_vio(2, xyz=(1, 2, 3)))
    assert mapped is not None
    assert alignment.state == "tracking"
    np.testing.assert_array_equal(mapped.T_map_device[:3, 3], [11, 22, 33])


def test_historical_good_fix_during_quality_loss_does_not_report_tracking(make_vio):
    alignment = PoseAlignment()
    alignment.add_vio(make_vio(0))
    alignment.add_vio(make_vio(10))
    alignment.add_vio(make_vio(20, quality_good=False))
    assert alignment.add_fix(5, np.eye(4), "session-A")
    assert alignment.state == "lost"
    assert alignment.add_vio(make_vio(21)) is not None
    assert alignment.state == "tracking"


@pytest.mark.parametrize("quality_good", [True, False])
def test_session_change_clears_alignment_even_with_bad_quality(make_vio, quality_good):
    alignment = PoseAlignment()
    alignment.add_vio(make_vio(100))
    assert alignment.add_fix(100, np.eye(4), "session-A")
    new_sample = make_vio(0, session_uid="session-B", quality_good=quality_good)
    assert alignment.add_vio(new_sample) is None
    assert alignment.state == "lost"
    assert not alignment.add_fix(100, np.eye(4), "session-A")
    assert not alignment.add_fix(100, np.eye(4), "session-B")
    assert alignment.add_vio(make_vio(101)) is None
    assert alignment.state == "lost"
    assert alignment.add_vio(make_vio(1, session_uid="session-B")) is None
    assert alignment.add_fix(1, np.eye(4), "session-B")
    mapped = alignment.add_vio(make_vio(2, session_uid="session-B"))
    assert mapped is not None
    assert mapped.session_uid == "session-B"
    assert mapped.fix_timestamp_ns == 1


def test_fix_cannot_establish_or_switch_session(make_vio):
    alignment = PoseAlignment()
    assert not alignment.add_fix(0, np.eye(4), "session-B")
    alignment.add_vio(make_vio(0))
    assert not alignment.add_fix(0, np.eye(4), "session-B")
    assert alignment.add_fix(0, np.eye(4), "session-A")
    assert alignment.add_vio(make_vio(1)) is not None


@pytest.mark.parametrize("timestamp_ns", [9, 10])
def test_out_of_order_and_duplicate_vio_are_ignored_without_losing_state(
    make_vio, timestamp_ns
):
    alignment = PoseAlignment()
    alignment.add_vio(make_vio(10))
    assert alignment.add_fix(10, np.eye(4), "session-A")
    assert alignment.add_vio(make_vio(timestamp_ns, quality_good=False)) is None
    assert alignment.state == "tracking"
    assert alignment.add_vio(make_vio(timestamp_ns, xyz=(999, 999, 999))) is None
    mapped = alignment.add_vio(make_vio(11))
    assert mapped is not None
    np.testing.assert_array_equal(mapped.T_map_device, np.eye(4))


@pytest.mark.parametrize("fix_timestamp_ns", [0, 10])
def test_out_of_order_fixes_do_not_overwrite_latest_alignment(
    make_vio, fix_timestamp_ns
):
    alignment = PoseAlignment()
    alignment.add_vio(make_vio(0))
    alignment.add_vio(make_vio(10))
    assert alignment.add_fix(10, np.eye(4), "session-A")
    wrong_pose = np.eye(4)
    wrong_pose[0, 3] = 100
    assert not alignment.add_fix(fix_timestamp_ns, wrong_pose, "session-A")
    mapped = alignment.add_vio(make_vio(11))
    assert mapped is not None
    assert mapped.fix_timestamp_ns == 10
    np.testing.assert_array_equal(mapped.T_map_device, np.eye(4))


def test_new_fix_replaces_alignment_without_averaging(make_vio, make_view):
    alignment = PoseAlignment()
    alignment.add_vio(make_vio(0))
    assert alignment.add_fix(0, np.eye(4), "session-A")
    alignment.add_vio(make_vio(1))
    new_map_pose = make_view(xyz=(100, 20, 30), yaw_deg=90).T_world_camera
    assert alignment.add_fix(1, new_map_pose, "session-A")
    mapped = alignment.add_vio(make_vio(2))
    assert mapped is not None
    np.testing.assert_array_equal(mapped.T_map_device, new_map_pose)
    assert mapped.fix_timestamp_ns == 1


def test_freshness_120_second_boundary_and_new_fix_recovery(make_vio):
    alignment = PoseAlignment()
    alignment.add_vio(make_vio(0))
    assert alignment.add_fix(0, np.eye(4), "session-A")
    fresh = alignment.add_vio(make_vio(120_000_000_000))
    assert fresh is not None
    assert fresh.age_ns == 120_000_000_000
    assert fresh.status == alignment.state == "tracking"
    stale = alignment.add_vio(make_vio(120_000_000_001))
    assert stale is not None
    assert stale.status == alignment.state == "stale"
    assert stale.age_ns == 120_000_000_001
    assert alignment.add_fix(stale.timestamp_ns, np.eye(4), "session-A")
    assert alignment.state == "tracking"
    fresh_again = alignment.add_vio(make_vio(120_000_000_002))
    assert fresh_again is not None
    assert fresh_again.age_ns == 1
    assert fresh_again.status == "tracking"


def test_zero_fix_age_limit_marks_next_observation_stale(make_vio):
    alignment = PoseAlignment(max_fix_age_ns=0)
    alignment.add_vio(make_vio(0))
    assert alignment.add_fix(0, np.eye(4), "session-A")
    assert alignment.state == "tracking"
    mapped = alignment.add_vio(make_vio(1))
    assert mapped is not None
    assert mapped.status == alignment.state == "stale"


def test_buffer_duration_is_inclusive_and_evicted_fixes_rejected(make_vio):
    alignment = PoseAlignment(buffer_duration_ns=5)
    alignment.add_vio(make_vio(0))
    alignment.add_vio(make_vio(5))
    assert alignment.add_fix(0, np.eye(4), "session-A")
    assert alignment.add_vio(make_vio(6)) is not None
    assert not alignment.add_fix(1, np.eye(4), "session-A")
    assert alignment.add_fix(5, np.eye(4), "session-A")


def test_sample_count_is_a_hard_bound_even_at_high_frequency(make_vio):
    alignment = PoseAlignment(max_samples=2)
    for timestamp_ns in (0, 1, 2):
        assert alignment.add_vio(make_vio(timestamp_ns)) is None
    assert not alignment.add_fix(0, np.eye(4), "session-A")
    assert alignment.add_fix(1, np.eye(4), "session-A")


def test_default_window_retains_capture_through_ten_second_solve(make_vio):
    alignment = PoseAlignment()
    alignment.add_vio(make_vio(0))
    alignment.add_vio(make_vio(10_000_000_000))
    assert alignment.add_fix(0, np.eye(4), "session-A")
    alignment.add_vio(make_vio(30_000_000_000))
    assert alignment.add_fix(10_000_000_000, np.eye(4), "session-A")


def test_input_and_output_matrices_cannot_mutate_internal_alignment(make_vio):
    alignment = PoseAlignment()
    sample = make_vio(0, xyz=(1, 2, 3))
    alignment.add_vio(sample)
    sample.T_odometry_device[:3, 3] = [999, 999, 999]
    map_pose = np.eye(4)
    assert alignment.add_fix(0, map_pose, "session-A")
    map_pose[:3, 3] = [999, 999, 999]
    mapped = alignment.add_vio(make_vio(1, xyz=(1, 2, 3)))
    assert mapped is not None
    np.testing.assert_array_equal(mapped.T_map_device, np.eye(4))
    mapped.T_map_device[:3, 3] = [999, 999, 999]
    next_mapped = alignment.add_vio(make_vio(2, xyz=(1, 2, 3)))
    assert next_mapped is not None
    np.testing.assert_array_equal(next_mapped.T_map_device, np.eye(4))


@pytest.mark.parametrize("timestamp_ns", [-1, 1.5, True, "1", None])
def test_invalid_timestamps_raise_without_establishing_session(timestamp_ns):
    alignment = PoseAlignment()
    with pytest.raises(ValueError, match="timestamp_ns"):
        alignment.add_vio(VioSample(timestamp_ns, np.eye(4), "session-A"))
    with pytest.raises(ValueError, match="corrected_timestamp_ns"):
        alignment.add_fix(timestamp_ns, np.eye(4), "session-A")
    assert alignment.state == "lost"


def test_numpy_integer_timestamps_are_normalized_without_losing_precision(make_vio):
    alignment = PoseAlignment()
    alignment.add_vio(make_vio(np.int64(2**60)))
    assert alignment.add_fix(np.int64(2**60), np.eye(4), "session-A")
    mapped = alignment.add_vio(make_vio(np.int64(2**60 + 1)))
    assert mapped is not None
    assert type(mapped.timestamp_ns) is int
    assert type(mapped.fix_timestamp_ns) is int
    assert mapped.age_ns == 1


@pytest.mark.parametrize("session_uid", ["", " \t", None, 3])
def test_invalid_sessions_raise(session_uid):
    alignment = PoseAlignment()
    with pytest.raises(ValueError, match="session_uid"):
        alignment.add_vio(VioSample(0, np.eye(4), session_uid))
    with pytest.raises(ValueError, match="session_uid"):
        alignment.add_fix(0, np.eye(4), session_uid)
    assert alignment.state == "lost"


@pytest.mark.parametrize("quality_good", [0, 1, None])
def test_quality_must_be_explicit_boolean(quality_good):
    alignment = PoseAlignment()
    with pytest.raises(ValueError, match="quality_good"):
        alignment.add_vio(VioSample(0, np.eye(4), "session-A", quality_good))


@pytest.mark.parametrize("invalid_pose", [np.eye(3), np.full((4, 4), np.nan)])
def test_invalid_se3_at_public_boundary_is_rejected_atomically(make_vio, invalid_pose):
    alignment = PoseAlignment()
    alignment.add_vio(make_vio(0))
    assert alignment.add_fix(0, np.eye(4), "session-A")
    with pytest.raises(ValueError, match="transform"):
        alignment.add_vio(VioSample(1, invalid_pose, "session-A"))
    with pytest.raises(ValueError, match="transform"):
        alignment.add_fix(1, invalid_pose, "session-A")
    assert alignment.state == "tracking"
    mapped = alignment.add_vio(make_vio(1))
    assert mapped is not None
    np.testing.assert_array_equal(mapped.T_map_device, np.eye(4))


def test_invalid_new_session_pose_still_clears_old_frame(make_vio):
    alignment = PoseAlignment()
    alignment.add_vio(make_vio(0))
    assert alignment.add_fix(0, np.eye(4), "session-A")
    with pytest.raises(ValueError, match="transform"):
        alignment.add_vio(VioSample(0, np.eye(3), "session-B"))
    assert alignment.state == "lost"
    assert not alignment.add_fix(0, np.eye(4), "session-B")
    assert not alignment.add_fix(0, np.eye(4), "session-A")
    assert alignment.add_vio(make_vio(1)) is None
    assert alignment.add_vio(make_vio(0, session_uid="session-B")) is None
    assert alignment.add_fix(0, np.eye(4), "session-B")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"buffer_duration_ns": 0},
        {"buffer_duration_ns": -1},
        {"buffer_duration_ns": 1.5},
        {"max_samples": 1},
        {"max_samples": True},
        {"max_interpolation_gap_ns": 0},
        {"max_interpolation_gap_ns": 250_000_001},
        {"max_fix_age_ns": -1},
        {"max_fix_age_ns": 120_000_000_001},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PoseAlignment(**kwargs)
