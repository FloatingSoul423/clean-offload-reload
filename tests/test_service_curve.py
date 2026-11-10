import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

MODULE_PATH = Path(__file__).parents[1] / "controller" / "swe_replay.py"
SPEC = importlib.util.spec_from_file_location("service_curve_replay", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def state(
    session_id, deadline, total, max_missing, applied_target=None, committed=None
):
    return {
        "session_id": session_id,
        "conditional_epochs": 0,
        "deadline": deadline,
        "total_bytes": total,
        "committed_missing_bytes": (
            max_missing if committed is None else committed
        ),
        "max_missing_bytes": max_missing,
        "applied_target_bytes": (
            total - max_missing if applied_target is None else applied_target
        ),
    }


def make_policy(**overrides):
    args = {
        "base": "http://127.0.0.1:30000",
        "stop": threading.Event(),
        "mean": 3.0,
        "sigma": 0.8,
        "interval": 0.05,
        "quantile": 0.10,
        "bytes_per_token": 1,
        "shared_gbps": 8.0,
        "safety": 0.9,
        "observe_bandwidth": False,
        "observe_hbm": False,
        "reserve_high": 0.15,
        "max_conditional_epochs": 1,
        "prefetch_lead_s": 0.0,
    }
    args.update(overrides)
    return MODULE.ServiceCurveEDFPolicy(**args)


class ServiceCurvePlannerTests(unittest.TestCase):
    def test_no_hbm_need_admits_no_new_eviction(self):
        states = {
            "far": state("far", 20.0, 100.0, 100.0, committed=0.0),
        }
        targets, _, _ = MODULE.plan_service_curve_targets(
            states,
            now=10.0,
            bandwidth_bytes_s=100.0,
            safety=1.0,
            eviction_need_bytes=0.0,
        )
        self.assertEqual(targets["far"], 100.0)

    def test_hbm_need_is_partial_and_far_deadline_first(self):
        states = {
            "near": state("near", 11.0, 100.0, 100.0, committed=0.0),
            "far": state("far", 12.0, 100.0, 100.0, committed=0.0),
        }
        targets, _, _ = MODULE.plan_service_curve_targets(
            states,
            now=10.0,
            bandwidth_bytes_s=1000.0,
            safety=1.0,
            eviction_need_bytes=60.0,
        )
        self.assertEqual(targets["far"], 40.0)
        self.assertEqual(targets["near"], 100.0)

    def test_far_deadline_first_fills_exact_prefix_slack(self):
        states = {
            "near": state("near", 11.0, 2.0, 2.0, committed=0.0),
            "far": state("far", 12.0, 2.0, 1.4, committed=0.0),
        }
        targets, demand, capacity = MODULE.plan_service_curve_targets(
            states, now=10.0, bandwidth_bytes_s=1.0, safety=1.0
        )
        self.assertAlmostEqual(targets["far"], 0.6)
        self.assertAlmostEqual(targets["near"], 1.4)
        self.assertAlmostEqual(demand[11.0], 0.6)
        self.assertAlmostEqual(demand[12.0], capacity[12.0])

    def test_saturated_curve_keeps_new_wait_fully_resident(self):
        states = {
            "existing": state("existing", 11.0, 1.0, 1.0),
            "new": state(
                "new", 11.0, 2.0, 2.0, applied_target=2.0, committed=0.0
            ),
        }
        targets, _, _ = MODULE.plan_service_curve_targets(
            states, now=10.0, bandwidth_bytes_s=1.0, safety=1.0
        )
        self.assertEqual(targets["new"], 2.0)

    def test_time_advance_requires_only_the_feasibility_reload(self):
        states = {"s": state("s", 11.0, 2.0, 0.8)}
        targets, _, _ = MODULE.plan_service_curve_targets(
            states, now=10.5, bandwidth_bytes_s=1.0, safety=1.0
        )
        self.assertAlmostEqual(targets["s"], 1.5)

    def test_single_wait_can_use_the_whole_shared_link(self):
        states = {"s": state("s", 11.0, 4.0, 4.0, committed=0.0)}
        targets, _, _ = MODULE.plan_service_curve_targets(
            states, now=10.0, bandwidth_bytes_s=8.0, safety=0.9
        )
        self.assertEqual(targets["s"], 0.0)

    def test_utilization_is_maximum_edf_prefix_ratio(self):
        states = {
            "near": state("near", 11.0, 1.0, 0.45),
            "far": state("far", 12.0, 1.0, 0.45),
        }
        utilization = MODULE.service_curve_utilization(
            states, now=10.0, bandwidth_bytes_s=1.0, safety=0.9
        )
        self.assertAlmostEqual(utilization, 0.5)


class ObservedServiceRateTests(unittest.TestCase):
    def test_parses_runtime_load_back_metrics(self):
        payload = b"""\
# HELP sglang:load_back_bytes_total bytes
sglang:load_back_bytes_total{rank=\"0\"} 4096
sglang:load_back_duration_seconds_sum{rank=\"0\"} 0.25
sglang:load_back_duration_seconds_count{rank=\"0\"} 2
sglang:hicache_backup_bytes_total{rank=\"0\"} 9999
"""

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return payload

        with patch.object(MODULE.urllib.request, "urlopen", return_value=Response()):
            self.assertEqual(MODULE.read_c2g_counters("base"), (4096.0, 0.25))

    def test_uses_median_recent_rate_and_clamps_to_ceiling(self):
        rate = MODULE.ObservedServiceRate(
            "http://127.0.0.1:30000", 8 * MODULE.GB, poll_s=0.5, window=5
        )
        counters = [
            (0.0, 0.0),
            (2 * MODULE.GB, 0.5),
            (12 * MODULE.GB, 1.0),
        ]
        with patch.object(MODULE, "read_c2g_counters", side_effect=counters):
            self.assertEqual(rate.update(0.0), 8 * MODULE.GB)
            self.assertEqual(rate.update(0.5), 4 * MODULE.GB)
            self.assertEqual(rate.update(1.0), 6 * MODULE.GB)

    def test_metric_failure_keeps_last_safe_rate(self):
        rate = MODULE.ObservedServiceRate("base", 8 * MODULE.GB)
        with patch.object(MODULE, "read_c2g_counters", side_effect=OSError):
            self.assertEqual(rate.update(0.0), 8 * MODULE.GB)
        self.assertEqual(rate.sample_errors, 1)


class ObservedHBMPressureTests(unittest.TestCase):
    def test_parses_destination_capacity_and_allocation_pressure(self):
        payload = b"""\
sglang:kv_available_tokens{rank=\"0\"} 3000
sglang:max_total_num_tokens{rank=\"0\"} 10000
sglang:pending_prealloc_token_usage{rank=\"0\"} 0.05
sglang:num_queue_reqs{rank=\"0\",type=\"regular\"} 2
"""

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return payload

        with patch.object(MODULE.urllib.request, "urlopen", return_value=Response()):
            sample = MODULE.read_hbm_pressure("base")
        self.assertEqual(sample["available_tokens"], 3000)
        self.assertEqual(sample["capacity_tokens"], 10000)
        self.assertEqual(sample["pending_prealloc_usage"], 0.05)
        self.assertEqual(sample["queued_requests"], 2)


class ServiceCurvePolicyTests(unittest.TestCase):
    def test_reload_can_use_reserve_but_not_pending_allocation(self):
        policy = make_policy(observe_hbm=True, bytes_per_token=1)
        policy.hbm.update = Mock(return_value={
            "available_tokens": 100.0,
            "capacity_tokens": 1000.0,
            "pending_prealloc_usage": 0.02,
            "queued_requests": 0.0,
        })
        need, headroom, _ = policy._hbm_budget(10.0)
        self.assertEqual(need, 0.0)
        self.assertEqual(headroom, 30.0)

    def test_reserve_uses_low_high_hysteresis(self):
        policy = make_policy(observe_hbm=True, bytes_per_token=1)
        samples = [
            {"available_tokens": 40.0, "capacity_tokens": 1000.0,
             "pending_prealloc_usage": 0.0, "queued_requests": 0.0},
            {"available_tokens": 100.0, "capacity_tokens": 1000.0,
             "pending_prealloc_usage": 0.0, "queued_requests": 0.0},
            {"available_tokens": 160.0, "capacity_tokens": 1000.0,
             "pending_prealloc_usage": 0.0, "queued_requests": 0.0},
        ]
        policy.hbm.update = Mock(side_effect=samples)
        self.assertEqual(policy._hbm_budget(1.0)[0], 110.0)
        self.assertEqual(policy._hbm_budget(2.0)[0], 50.0)
        self.assertEqual(policy._hbm_budget(3.0)[0], 0.0)
        self.assertFalse(policy.reserve_active)

    def test_real_hbm_need_gates_and_partially_sizes_eviction(self):
        policy = make_policy(
            observe_hbm=True,
            bytes_per_token=1,
            shared_gbps=1.0,
            reserve_high=0.15,
        )
        policy.hbm.update = Mock(return_value={
            "available_tokens": 40.0,
            "capacity_tokens": 1000.0,
            "pending_prealloc_usage": 0.0,
            "queued_requests": 0.0,
        })
        sent = []
        with patch.object(MODULE.time, "monotonic", return_value=10.0), patch.object(
            MODULE, "notify_tool_probability",
            side_effect=lambda base, sid, target: sent.append(target) or True,
        ):
            policy.begin("s", 200)
        self.assertEqual(sent, [0.45])
        self.assertEqual(policy.waiting["s"]["committed_missing_bytes"], 110.0)

    def test_no_space_need_sends_no_eviction(self):
        policy = make_policy(observe_hbm=True, bytes_per_token=1)
        policy.hbm.update = Mock(return_value={
            "available_tokens": 200.0,
            "capacity_tokens": 1000.0,
            "pending_prealloc_usage": 0.0,
            "queued_requests": 0.0,
        })
        with patch.object(MODULE.time, "monotonic", return_value=10.0), patch.object(
            MODULE, "notify_tool_probability"
        ) as notify:
            policy.begin("s", 100)
        notify.assert_not_called()

    def test_surviving_full_session_opens_conditional_deadline_epoch(self):
        policy = make_policy(observe_hbm=True, bytes_per_token=1)
        policy.hbm.update = Mock(return_value={
            "available_tokens": 0.0,
            "capacity_tokens": 1000.0,
            "pending_prealloc_usage": 0.0,
            "queued_requests": 0.0,
        })
        policy.waiting["s"] = {
            **state("s", 15.0, 100.0, 0.0, applied_target=100.0),
            "wait_started": 10.0,
        }
        with patch.object(MODULE.time, "monotonic", return_value=20.0), patch.object(
            MODULE, "notify_tool_probability", return_value=True
        ):
            policy.rebalance(force=True)
        self.assertGreater(policy.waiting["s"]["deadline"], 20.0)
        self.assertEqual(policy.stats["deadline_epoch_rolls"], 1)

    def test_wait_can_open_only_one_conditional_epoch(self):
        policy = make_policy(
            observe_hbm=True,
            bytes_per_token=1,
            max_conditional_epochs=1,
        )
        policy.hbm.update = Mock(return_value={
            "available_tokens": 0.0,
            "capacity_tokens": 1000.0,
            "pending_prealloc_usage": 0.0,
            "queued_requests": 0.0,
        })
        policy.waiting["s"] = {
            **state("s", 15.0, 100.0, 0.0, applied_target=100.0),
            "wait_started": 10.0,
            "conditional_epochs": 1,
        }
        with patch.object(MODULE.time, "monotonic", return_value=30.0), patch.object(
            MODULE, "notify_tool_probability", return_value=True
        ):
            policy.rebalance(force=True)
        self.assertEqual(policy.waiting["s"]["deadline"], 15.0)
        self.assertEqual(policy.stats["deadline_epoch_rolls"], 0)

    def test_global_reload_cooldown_issues_only_one_chunk(self):
        policy = make_policy(
            bytes_per_token=1,
            reload_chunk_gb=10.0 / MODULE.GB,
            reload_cooldown_s=0.2,
        )
        for sid in ("a", "b"):
            policy.waiting[sid] = {
                **state(sid, 10.0, 100.0, 100.0, applied_target=0.0),
                "wait_started": 5.0,
            }
        sent = []
        with patch.object(MODULE.time, "monotonic", return_value=10.0), patch.object(
            MODULE, "notify_tool_probability",
            side_effect=lambda base, sid, target: sent.append((sid, target)) or True,
        ):
            policy.rebalance(force=True)
        self.assertEqual(len(sent), 1)

    def test_reload_is_chunked(self):
        policy = make_policy(
            bytes_per_token=1,
            reload_chunk_gb=10.0 / MODULE.GB,
        )
        policy.waiting["s"] = {
            **state("s", 10.0, 100.0, 100.0, applied_target=0.0),
            "wait_started": 5.0,
        }
        sent = []
        with patch.object(MODULE.time, "monotonic", return_value=10.0), patch.object(
            MODULE, "notify_tool_probability",
            side_effect=lambda base, sid, target: sent.append(target) or True,
        ):
            policy.rebalance(force=True)
        self.assertEqual(sent, [0.1])
        self.assertEqual(policy.waiting["s"]["applied_target_bytes"], 10.0)

    def test_prefetch_lead_starts_edf_reload_early(self):
        policy = make_policy(
            bytes_per_token=1,
            shared_gbps=1.0,
            reload_chunk_gb=10.0,
            prefetch_lead_s=1.0,
        )
        policy.waiting["s"] = {
            **state(
                "s",
                12.0,
                2 * MODULE.GB,
                2 * MODULE.GB,
                applied_target=0.0,
            ),
            "wait_started": 5.0,
        }
        sent = []
        with patch.object(MODULE.time, "monotonic", return_value=10.0), patch.object(
            MODULE, "notify_tool_probability",
            side_effect=lambda base, sid, target: sent.append(target) or True,
        ):
            policy.rebalance(force=True)
        self.assertEqual(len(sent), 1)
        self.assertAlmostEqual(sent[0], 0.55)

    def test_idle_headroom_is_used_for_opportunistic_reload(self):
        policy = make_policy(
            observe_hbm=True,
            bytes_per_token=1,
            reload_chunk_gb=10.0 / MODULE.GB,
        )
        policy.hbm.update = Mock(return_value={
            "available_tokens": 200.0,
            "capacity_tokens": 1000.0,
            "pending_prealloc_usage": 0.0,
            "queued_requests": 0.0,
        })
        policy.waiting["s"] = {
            **state("s", 20.0, 100.0, 50.0, applied_target=50.0),
            "wait_started": 5.0,
        }
        sent = []
        with patch.object(MODULE.time, "monotonic", return_value=10.0), patch.object(
            MODULE, "notify_tool_probability",
            side_effect=lambda base, sid, target: sent.append(target) or True,
        ):
            policy.rebalance(force=True)
        self.assertEqual(sent, [0.6])

    def test_first_target_is_curve_admission_not_zero_then_reload(self):
        policy = make_policy(shared_gbps=1.0)
        policy.deadline_offset = 1.0
        sent = []
        with patch.object(MODULE.time, "monotonic", return_value=10.0), patch.object(
            MODULE, "notify_tool_probability",
            side_effect=lambda base, sid, target: sent.append(target) or True,
        ):
            policy.begin("s", 2 * MODULE.GB)
        self.assertEqual(len(sent), 1)
        self.assertAlmostEqual(sent[0], 0.55)
        self.assertAlmostEqual(
            policy.waiting["s"]["committed_missing_bytes"],
            0.9 * MODULE.GB,
            delta=1e-3,
        )
        self.assertEqual(
            policy.waiting["s"]["max_missing_bytes"], 2 * MODULE.GB
        )

    def test_accepted_target_cannot_be_re_evicted_later(self):
        policy = make_policy(shared_gbps=1.0)
        policy.waiting["s"] = state(
            "s", 20.0, 2 * MODULE.GB, 0.5 * MODULE.GB
        )
        with patch.object(MODULE.time, "monotonic", return_value=10.0), patch.object(
            MODULE, "notify_tool_probability"
        ) as notify:
            policy.rebalance(force=True)
        notify.assert_not_called()
        self.assertEqual(
            policy.waiting["s"]["applied_target_bytes"], 1.5 * MODULE.GB
        )

    def test_finish_requests_full_residency(self):
        policy = make_policy()
        policy.waiting["s"] = state("s", 20.0, 100.0, 50.0)
        with patch.object(
            MODULE, "notify_tool_probability", return_value=True
        ) as notify:
            policy.finish("s")
        notify.assert_called_once_with(policy.base, "s", 1.0)

    def test_wait_propagates_interrupt_and_cleans_state(self):
        policy = make_policy()
        policy.begin = Mock()
        policy.cancel = Mock()
        policy.stop = Mock()
        policy.stop.wait.return_value = True
        self.assertTrue(policy.wait("s", 2.0, 4000))
        policy.cancel.assert_called_once_with("s")


if __name__ == "__main__":
    unittest.main()
