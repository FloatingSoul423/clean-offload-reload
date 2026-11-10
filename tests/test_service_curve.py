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
    }
    args.update(overrides)
    return MODULE.ServiceCurveEDFPolicy(**args)


class ServiceCurvePlannerTests(unittest.TestCase):
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


class ServiceCurvePolicyTests(unittest.TestCase):
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
            policy.waiting["s"]["max_missing_bytes"],
            0.9 * MODULE.GB,
            delta=1e-3,
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
