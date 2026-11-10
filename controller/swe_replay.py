#!/usr/bin/env python3
"""Replay SWE-agent turn lengths against SGLang with real SWE-bench issue text.

The turn lengths come from a downloaded SWE-agent trajectory summary. Issue
statements come from the local SWE-bench dataset. Tool observations are synthetic
text of matched length, and inter-arrival times are sampled. This probes the
real model server and KV-cache transfer path, but is not an agent execution.
"""
import argparse
import json
import math
import os
import random
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

GB = 1 << 30
KiB = 1 << 10
TRACES = os.path.expanduser(
    "~/.cache/clean-offload-reload/swe_traces.json"
)


def build_traces(model_dir, out=TRACES, n=40):
    """Extract (turn -> observation tokens, context before) from the dataset."""
    if os.path.exists(out):
        with open(out) as f:
            return json.load(f)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from datasets import load_dataset
    tok = get_tok(model_dir)
    d = load_dataset("nebius/SWE-agent-trajectories", split="train",
                     streaming=True)
    sess = []
    for i, r in enumerate(d):
        if i >= n:
            break
        turns, ctx = [], 0
        for m in r["trajectory"]:
            txt = m.get("text") or m.get("system_prompt") or ""
            k = len(tok(txt, add_special_tokens=False)["input_ids"]) if txt else 0
            if m.get("role") == "user":
                turns.append({"obs": k, "ctx_before": ctx})
            ctx += k
        if turns:
            sess.append({"turns": turns, "total_ctx": ctx})
    Path(out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(sess, f)
    return sess


_TOK = None


def get_tok(d):
    """transformers 5.x removed AutoTokenizer from the top-level namespace;
    import from the submodule, falling back to the old path for 4.x."""
    global _TOK
    if _TOK is None:
        try:
            from transformers.models.auto.tokenization_auto import AutoTokenizer
        except ImportError:
            from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(d)
    return _TOK


def get_model_config(model):
    try:
        from transformers.models.auto.configuration_auto import AutoConfig
    except ImportError:
        from transformers import AutoConfig
    return AutoConfig.from_pretrained(model).to_dict()


def filler(tag, ntok, model_dir):
    """Exactly ntok tokens, unique to tag so sessions never share a prefix."""
    if ntok <= 0:
        return ""
    tok = get_tok(model_dir)
    seed = abs(hash(tag)) % 100000
    w = [f"item{(seed + i * 7919) % 99991}" for i in range(ntok + 128)]
    ids = tok(" ".join(w), add_special_tokens=False)["input_ids"]
    while len(ids) < ntok:
        w += [f"item{(seed + j) % 99991}" for j in range(256)]
        ids = tok(" ".join(w), add_special_tokens=False)["input_ids"]
    return tok.decode(ids[:ntok])


def stream_ttft(url, payload, timeout=900):
    body = json.dumps(dict(payload, stream=True)).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, out = None, []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            if not raw.startswith(b"data:"):
                continue
            c = raw[5:].strip()
            if c == b"[DONE]":
                break
            try:
                j = json.loads(c)
            except Exception:
                continue
            ch = j.get("choices", [{}])[0]
            t = ch.get("text") or ch.get("delta", {}).get("content")
            if t:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                out.append(t)
    return ttft, time.perf_counter() - t0, "".join(out)


def tool_return_cdf(elapsed, mean, sigma):
    """CDF of the same lognormal distribution used for synthetic tool time."""
    if elapsed <= 0:
        return 0.0
    mu = math.log(mean) - sigma * sigma / 2
    z = (math.log(elapsed) - mu) / (sigma * math.sqrt(2))
    return min(1.0, max(0.0, 0.5 * (1.0 + math.erf(z))))


def notify_tool_probability(base, session_id, probability):
    body = json.dumps({
        "session_id": session_id,
        "return_probability": probability,
    }).encode()
    req = urllib.request.Request(
        base + "/tool_cache_probability",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        return response.status == 202


def scrape_token_usage(base):
    """Read the real SGLang HBM KV-pool pressure gauge."""
    values = []
    with urllib.request.urlopen(base + "/metrics", timeout=3) as response:
        for line in response.read().decode().splitlines():
            if not (
                line.startswith("sglang:token_usage ")
                or line.startswith("sglang:token_usage{")
            ):
                continue
            try:
                values.append(float(line.rsplit(None, 1)[1]))
            except ValueError:
                continue
    return max(values) if values else None


def wait_with_probability(
    stop, duration, base, session_id, mean, sigma, update_interval
):
    """Actively raise HBM residency while the tool is still running."""
    started = time.monotonic()
    notify_tool_probability(base, session_id, 0.0)
    while True:
        elapsed = time.monotonic() - started
        remaining = duration - elapsed
        if remaining <= 0:
            notify_tool_probability(base, session_id, 1.0)
            return False
        if stop.wait(min(update_interval, remaining)):
            return True
        elapsed = time.monotonic() - started
        notify_tool_probability(
            base,
            session_id,
            tool_return_cdf(elapsed, mean, sigma),
        )


def conditional_quantile_remaining(elapsed, mean, sigma, quantile):
    """Return the conditional lognormal quantile given that the tool still runs.

    ``quantile=0.10`` means that, among calls still running at ``elapsed``, ten
    percent are predicted to finish within the returned number of seconds.
    """
    if elapsed < 0:
        raise ValueError("elapsed must be non-negative")
    if mean <= 0 or sigma <= 0:
        raise ValueError("mean and sigma must be positive")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    current_cdf = tool_return_cdf(elapsed, mean, sigma)
    target_cdf = current_cdf + (1.0 - current_cdf) * quantile
    z = statistics.NormalDist().inv_cdf(min(1.0 - 1e-9, target_cdf))
    mu = math.log(mean) - sigma * sigma / 2
    return max(0.0, math.exp(mu + sigma * z) - elapsed)


def lognormal_quantile(mean, sigma, quantile):
    """Return one absolute tool-duration quantile fixed at call start."""
    if mean <= 0 or sigma <= 0:
        raise ValueError("mean and sigma must be positive")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    mu = math.log(mean) - sigma * sigma / 2
    return math.exp(mu + sigma * statistics.NormalDist().inv_cdf(quantile))


def deadline_residency_fraction(
    *, total_bytes, seconds_to_deadline, bandwidth_bytes_s, safety=0.9
):
    """Minimum HBM fraction needed now to finish Reload by the deadline."""
    if total_bytes <= 0:
        return 0.0
    if seconds_to_deadline < 0 or bandwidth_bytes_s <= 0:
        raise ValueError("deadline slack and bandwidth must be non-negative")
    if not 0.0 < safety <= 1.0:
        raise ValueError("safety must be in (0, 1]")
    reloadable = safety * bandwidth_bytes_s * seconds_to_deadline
    return min(1.0, max(0.0, 1.0 - reloadable / total_bytes))


def wait_with_deadline_residency(
    stop,
    duration,
    base,
    session_id,
    mean,
    sigma,
    update_interval,
    quantile,
    total_bytes,
    bandwidth_gbps,
):
    """Monotonically prepare KV as the conditional return deadline approaches.

    The existing endpoint accepts a fraction and is reused as the transport;
    unlike the old policy, that fraction is derived only from deadline slack
    and Reload bandwidth, not from return probability.
    """
    started = time.monotonic()
    absolute_deadline = lognormal_quantile(mean, sigma, quantile)
    target_fraction = 0.0
    notify_tool_probability(base, session_id, target_fraction)
    bandwidth_bytes_s = bandwidth_gbps * GB
    while True:
        elapsed = time.monotonic() - started
        remaining = duration - elapsed
        if remaining <= 0:
            notify_tool_probability(base, session_id, 1.0)
            return False
        if stop.wait(min(update_interval, remaining)):
            return True
        elapsed = time.monotonic() - started
        deadline_slack = max(0.0, absolute_deadline - elapsed)
        next_fraction = deadline_residency_fraction(
            total_bytes=total_bytes,
            seconds_to_deadline=deadline_slack,
            bandwidth_bytes_s=bandwidth_bytes_s,
        )
        # A later conditional estimate must not undo already prepared KV.
        target_fraction = max(target_fraction, next_fraction)
        notify_tool_probability(base, session_id, target_fraction)


class SharedBandwidthCoordinator:
    """Coordinate shared bandwidth and an optional one-victim pressure valve."""

    def __init__(
        self, per_request_gbps, shared_gbps, *, pressure_enabled=False,
        pressure_high=0.85, pressure_low=0.70, pressure_poll_s=0.20,
        recover_cycles=3, deadline_guard_s=0.10, cooldown_s=1.0,
        mean=3.0, sigma=0.8, quantile=0.10,
    ):
        self.per_request_bytes_s = per_request_gbps * GB
        self.shared_bytes_s = shared_gbps * GB
        self.per_request_gbps = per_request_gbps
        self.shared_gbps = shared_gbps
        self.lock = threading.Lock()
        self.waiting = {}
        self.pressure_enabled = pressure_enabled
        self.pressure_high = pressure_high
        self.pressure_low = pressure_low
        self.pressure_poll_s = pressure_poll_s
        self.recover_cycles = recover_cycles
        self.deadline_guard_s = deadline_guard_s
        self.cooldown_s = cooldown_s
        self.mean = mean
        self.sigma = sigma
        self.quantile = quantile
        self.last_pressure_poll = 0.0
        self.pressure_mode = False
        self.recover_strikes = 0
        self.victim_id = None
        self.stats = {
            "calls": 0,
            "samples": 0,
            "target_updates": 0,
            "bandwidth_limited_samples": 0,
            "max_waiting": 0,
            "max_parallel_reload_plans": 0,
            "effective_bandwidth_samples_gbps": 0.0,
            "pressure_polls": 0,
            "pressure_scrape_errors": 0,
            "pressure_entries": 0,
            "pressure_exits": 0,
            "pressure_samples_high": 0,
            "peak_token_usage": 0.0,
            "victim_selections": 0,
            "distinct_victim_calls": 0,
            "active_evict_transitions": 0,
            "readmit_transitions": 0,
            "victim_finishes": 0,
            "deadline_guard_skips": 0,
            "cooldown_skips": 0,
            "pressure_polls_without_safe_victim": 0,
            "active_reclaims_planned_bytes": 0,
        }

    def enter(self, session_id, total_bytes=0):
        with self.lock:
            self.waiting[session_id] = {
                "started": time.monotonic(),
                "total_bytes": total_bytes,
                "target_fraction": 0.0,
                "needs_reload": True,
                "sacrificed": False,
                "ever_sacrificed": False,
                "cooldown_until": 0.0,
            }
            self.stats["calls"] += 1
            self.stats["max_waiting"] = max(
                self.stats["max_waiting"], len(self.waiting)
            )

    def exit(self, session_id):
        with self.lock:
            if self.victim_id == session_id:
                self.victim_id = None
            self.waiting.pop(session_id, None)

    def effective_bandwidth(self, session_id, target_fraction):
        with self.lock:
            state = self.waiting.get(session_id)
            if state is not None:
                state["target_fraction"] = target_fraction
                state["needs_reload"] = (
                    not state["sacrificed"]
                    and target_fraction < 1.0 - 1e-6
                )
            active = sum(
                state["needs_reload"] for state in self.waiting.values()
            )
            fair_share = self.shared_bytes_s / max(1, active)
            effective = min(self.per_request_bytes_s, fair_share)
            self.stats["samples"] += 1
            self.stats["bandwidth_limited_samples"] += int(
                effective < self.per_request_bytes_s
            )
            self.stats["max_parallel_reload_plans"] = max(
                self.stats["max_parallel_reload_plans"], active
            )
            self.stats["effective_bandwidth_samples_gbps"] += effective / GB
            return effective

    def update_target(self, session_id, target_fraction):
        with self.lock:
            state = self.waiting.get(session_id)
            if state is not None:
                state["target_fraction"] = target_fraction
                state["needs_reload"] = (
                    not state["sacrificed"]
                    and target_fraction < 1.0 - 1e-6
                )

    def maybe_poll_pressure(self, base):
        if not self.pressure_enabled:
            return
        now = time.monotonic()
        with self.lock:
            if now - self.last_pressure_poll < self.pressure_poll_s:
                return
            self.last_pressure_poll = now
        try:
            usage = scrape_token_usage(base)
        except Exception:
            with self.lock:
                self.stats["pressure_scrape_errors"] += 1
            return
        if usage is None:
            return

        with self.lock:
            self.stats["pressure_polls"] += 1
            self.stats["peak_token_usage"] = max(
                self.stats["peak_token_usage"], usage
            )
            if usage >= self.pressure_high:
                self.stats["pressure_samples_high"] += 1
                self.recover_strikes = 0
                if not self.pressure_mode:
                    self.pressure_mode = True
                    self.stats["pressure_entries"] += 1
            elif usage <= self.pressure_low:
                self.recover_strikes += 1
                if (
                    self.pressure_mode
                    and self.recover_strikes >= self.recover_cycles
                ):
                    self.pressure_mode = False
                    self.stats["pressure_exits"] += 1
                    if self.victim_id in self.waiting:
                        state = self.waiting[self.victim_id]
                        state["sacrificed"] = False
                        state["needs_reload"] = True
                        state["cooldown_until"] = now + self.cooldown_s
                        self.stats["readmit_transitions"] += 1
                    self.victim_id = None
            else:
                self.recover_strikes = 0

            if not (
                self.pressure_mode
                and usage >= self.pressure_high
                and self.victim_id is None
            ):
                return

            choices = []
            for session_id, state in self.waiting.items():
                if state["sacrificed"] or state["target_fraction"] <= 0:
                    continue
                remaining_deadline = conditional_quantile_remaining(
                    max(0.0, now - state["started"]),
                    self.mean,
                    self.sigma,
                    self.quantile,
                )
                if remaining_deadline <= self.deadline_guard_s:
                    self.stats["deadline_guard_skips"] += 1
                    continue
                if now < state["cooldown_until"]:
                    self.stats["cooldown_skips"] += 1
                    continue
                reclaimable = (
                    state["target_fraction"] * state["total_bytes"]
                )
                choices.append((reclaimable, remaining_deadline, session_id))
            if not choices:
                self.stats["pressure_polls_without_safe_victim"] += 1
                return

            reclaimable, _, victim_id = max(choices)
            state = self.waiting[victim_id]
            state["sacrificed"] = True
            state["needs_reload"] = False
            self.victim_id = victim_id
            self.stats["victim_selections"] += 1
            self.stats["active_reclaims_planned_bytes"] += int(reclaimable)
            if not state["ever_sacrificed"]:
                state["ever_sacrificed"] = True
                self.stats["distinct_victim_calls"] += 1

    def is_sacrificed(self, session_id):
        with self.lock:
            state = self.waiting.get(session_id)
            return bool(state and state["sacrificed"])

    def record_active_evict(self):
        with self.lock:
            self.stats["active_evict_transitions"] += 1

    def record_finish(self, session_id):
        with self.lock:
            state = self.waiting.get(session_id)
            if state is not None and state["sacrificed"]:
                self.stats["victim_finishes"] += 1

    def record_update(self):
        with self.lock:
            self.stats["target_updates"] += 1

    def snapshot_stats(self):
        with self.lock:
            result = dict(self.stats)
            result["per_request_bandwidth_gbps"] = self.per_request_gbps
            result["shared_bandwidth_gbps"] = self.shared_gbps
            result["waiting_at_end"] = len(self.waiting)
            result["pressure_enabled"] = self.pressure_enabled
            result["pressure_high"] = self.pressure_high
            result["pressure_low"] = self.pressure_low
            result["deadline_guard_s"] = self.deadline_guard_s
            result["cooldown_s"] = self.cooldown_s
        result["mean_effective_bandwidth_gbps"] = round(
            result["effective_bandwidth_samples_gbps"]
            / max(1, result["samples"]),
            4,
        )
        return result


def wait_with_shared_bandwidth_deadline(
    stop, duration, base, session_id, mean, sigma, update_interval,
    quantile, total_bytes, coordinator
):
    """Run the original 1.0 loop, reading only a shared concurrency count."""
    started = time.monotonic()
    absolute_deadline = lognormal_quantile(mean, sigma, quantile)
    target_fraction = 0.0
    notify_tool_probability(base, session_id, target_fraction)
    coordinator.enter(session_id, total_bytes)
    was_sacrificed = False
    try:
        while True:
            elapsed = time.monotonic() - started
            remaining = duration - elapsed
            if remaining <= 0:
                coordinator.record_finish(session_id)
                notify_tool_probability(base, session_id, 1.0)
                return False
            if stop.wait(min(update_interval, remaining)):
                return True
            coordinator.maybe_poll_pressure(base)
            sacrificed = coordinator.is_sacrificed(session_id)
            if sacrificed:
                if not was_sacrificed:
                    notify_tool_probability(base, session_id, 0.0)
                    target_fraction = 0.0
                    coordinator.update_target(session_id, 0.0)
                    coordinator.record_active_evict()
                was_sacrificed = True
                continue
            if was_sacrificed:
                # Re-enter the ordinary shared-bandwidth 3.0 planner from zero.
                target_fraction = 0.0
                was_sacrificed = False
            elapsed = time.monotonic() - started
            deadline_slack = max(0.0, absolute_deadline - elapsed)
            bandwidth_bytes_s = coordinator.effective_bandwidth(
                session_id, target_fraction
            )
            next_fraction = deadline_residency_fraction(
                total_bytes=total_bytes,
                seconds_to_deadline=deadline_slack,
                bandwidth_bytes_s=bandwidth_bytes_s,
            )
            next_fraction = max(target_fraction, next_fraction)
            if next_fraction > target_fraction + 1e-6:
                target_fraction = next_fraction
                notify_tool_probability(base, session_id, target_fraction)
                coordinator.update_target(session_id, target_fraction)
                coordinator.record_update()
    finally:
        coordinator.exit(session_id)


def allocate_near_deadline_targets(states, budget_bytes):
    """Allocate an HBM byte budget from the nearest deadline outward.

    ``states`` maps a session id to ``deadline`` and ``ideal_target_bytes``.
    The allocation may fully fund several near sessions, partially fund one
    boundary session, and leave farther sessions at zero.  A zero target is a
    full device eviction only: SGLang keeps the radix path and its host copy.
    """
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be non-negative")
    targets = {session_id: 0.0 for session_id in states}
    remaining = float(budget_bytes)
    for session_id, state in sorted(
        states.items(), key=lambda item: (item[1]["deadline"], item[0])
    ):
        ideal = float(state["ideal_target_bytes"])
        if ideal < 0:
            raise ValueError("ideal_target_bytes must be non-negative")
        target = min(ideal, remaining)
        targets[session_id] = target
        remaining -= target
        if remaining <= 0:
            break
    return targets


class RuntimeDeadlineFloorPolicy:
    """v8 client registers tool deadlines; runtime owns floor enforcement."""
    def __init__(self, base, stop, mean, sigma, interval, quantile, **kwargs):
        self.base, self.stop, self.interval = base, stop, interval
        self.deadline = lognormal_quantile(mean, sigma, quantile)
        self.calls = 0
        self.lock = threading.Lock()

    def send(self, sid, action):
        payload = {"session_id": sid, "return_probability": 1.0,
                   "two_phase_action": action}
        if action == "begin":
            payload["deadline_seconds"] = self.deadline
        request = urllib.request.Request(
            self.base+"/tool_cache_probability", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status != 202:
                raise RuntimeError("deadline-floor registration rejected")

    def wait(self, session_id, duration, ctx_tokens):
        with self.lock:
            self.calls += 1
        self.send(session_id, "begin")
        try:
            # Scheduler ticks the controller even without client polling.
            return self.stop.wait(duration)
        finally:
            self.send(session_id, "end")

    def snapshot_stats(self):
        return {"version": "8.1-idle-reserve", "calls": self.calls,
                "decisions": "server log: deadline-floor decision",
                "deadline_offset_s": self.deadline}


def wait_with_repaired_v3_deadline(
    stop, duration, base, session_id, mean, sigma, update_interval,
    quantile, total_bytes, coordinator
):
    """3.1 shared-bandwidth loop: demote at wait start, ramp to p10, resend.

    The target is re-sent every cycle, even when unchanged, so a Reload the
    server deferred (no free HBM, producer slot busy, H2D in flight) is
    retried on the next cycle instead of waiting for the fraction to rise.
    """
    started = time.monotonic()
    absolute_deadline = lognormal_quantile(mean, sigma, quantile)
    target_fraction = 0.0
    notify_tool_probability(base, session_id, target_fraction)
    coordinator.enter(session_id, total_bytes)
    try:
        while True:
            remaining = duration - (time.monotonic() - started)
            if remaining <= 0:
                notify_tool_probability(base, session_id, 1.0)
                return False
            if stop.wait(min(update_interval, remaining)):
                return True
            deadline_slack = max(
                0.0, absolute_deadline - (time.monotonic() - started)
            )
            bandwidth_bytes_s = coordinator.effective_bandwidth(
                session_id, target_fraction
            )
            next_fraction = deadline_residency_fraction(
                total_bytes=total_bytes,
                seconds_to_deadline=deadline_slack,
                bandwidth_bytes_s=bandwidth_bytes_s,
            )
            target_fraction = max(target_fraction, next_fraction)
            notify_tool_probability(base, session_id, target_fraction)
            coordinator.update_target(session_id, target_fraction)
            coordinator.record_update()
    finally:
        coordinator.exit(session_id)


def read_c2g_counters(base, timeout=1.0):
    """Return cumulative H2D bytes and copy-engine seconds from Prometheus."""
    totals = {"bytes": 0.0, "time": 0.0}
    found = set()
    with urllib.request.urlopen(base + "/metrics", timeout=timeout) as response:
        for line in response.read().decode().splitlines():
            if line.startswith("#"):
                continue
            name = line.split("{", 1)[0].split(None, 1)[0]
            key = {
                "sglang:load_back_bytes_total": "bytes",
                "sglang:load_back_duration_seconds_sum": "time",
            }.get(name)
            if key is None:
                continue
            try:
                totals[key] += float(line.rsplit(None, 1)[1])
                found.add(key)
            except ValueError:
                continue
    if found != {"bytes", "time"}:
        return None
    return totals["bytes"], totals["time"]


def read_hbm_pressure(base, timeout=1.0):
    """Read the destination KV-pool state used by the v9.1 HBM constraint."""
    wanted = {
        "sglang:kv_available_tokens",
        "sglang:max_total_num_tokens",
        "sglang:pending_prealloc_token_usage",
        "sglang:num_queue_reqs",
    }
    values = {name: [] for name in wanted}
    with urllib.request.urlopen(base + "/metrics", timeout=timeout) as response:
        for line in response.read().decode().splitlines():
            if line.startswith("#"):
                continue
            name = line.split("{", 1)[0].split(None, 1)[0]
            if name not in wanted:
                continue
            try:
                values[name].append(float(line.rsplit(None, 1)[1]))
            except ValueError:
                continue
    if not values["sglang:kv_available_tokens"] or not values[
        "sglang:max_total_num_tokens"
    ]:
        return None
    return {
        "available_tokens": max(values["sglang:kv_available_tokens"]),
        "capacity_tokens": max(values["sglang:max_total_num_tokens"]),
        "pending_prealloc_usage": max(
            values["sglang:pending_prealloc_token_usage"] or [0.0]
        ),
        "queued_requests": sum(values["sglang:num_queue_reqs"] or [0.0]),
    }


class ObservedHBMPressure:
    """Cache HBM pressure telemetry and fail closed for new eviction."""

    def __init__(self, base, poll_s=0.1):
        if poll_s <= 0:
            raise ValueError("HBM poll interval must be positive")
        self.base = base
        self.poll_s = poll_s
        self.last_poll = float("-inf")
        self.current = None
        self.samples = 0
        self.sample_errors = 0

    def update(self, now):
        if now - self.last_poll < self.poll_s:
            return self.current
        self.last_poll = now
        try:
            sample = read_hbm_pressure(self.base)
        except Exception:
            self.sample_errors += 1
            return self.current
        if sample is None:
            self.sample_errors += 1
            return self.current
        self.current = sample
        self.samples += 1
        return self.current

    def snapshot(self):
        return {
            "samples": self.samples,
            "sample_errors": self.sample_errors,
            "current": self.current,
        }

    def adjust_available_tokens(self, delta):
        """Account for an accepted target update before the next scrape."""
        if self.current is None:
            return
        self.current = dict(self.current)
        self.current["available_tokens"] = min(
            self.current["capacity_tokens"],
            max(0.0, self.current["available_tokens"] + delta),
        )


class ObservedServiceRate:
    """Estimate the usable single-link service rate from completed H2D copies."""

    def __init__(self, base, ceiling_bytes_s, poll_s=0.5, window=5):
        if ceiling_bytes_s <= 0 or poll_s <= 0 or window <= 0:
            raise ValueError("service-rate parameters must be positive")
        self.base = base
        self.ceiling_bytes_s = ceiling_bytes_s
        self.poll_s = poll_s
        self.window = window
        self.current_bytes_s = ceiling_bytes_s
        self.last_poll = float("-inf")
        self.last_counters = None
        self.samples = []
        self.sample_errors = 0

    def update(self, now):
        if now - self.last_poll < self.poll_s:
            return self.current_bytes_s
        self.last_poll = now
        try:
            counters = read_c2g_counters(self.base)
        except Exception:
            self.sample_errors += 1
            return self.current_bytes_s
        if counters is None:
            self.sample_errors += 1
            return self.current_bytes_s
        if self.last_counters is not None:
            delta_bytes = counters[0] - self.last_counters[0]
            delta_time = counters[1] - self.last_counters[1]
            if delta_bytes > 0 and delta_time > 0:
                observed = min(self.ceiling_bytes_s, delta_bytes / delta_time)
                self.samples.append(observed)
                self.samples = self.samples[-self.window:]
                self.current_bytes_s = min(
                    self.ceiling_bytes_s, statistics.median(self.samples)
                )
        self.last_counters = counters
        return self.current_bytes_s

    def snapshot(self):
        return {
            "configured_ceiling_bytes_s": self.ceiling_bytes_s,
            "current_bytes_s": self.current_bytes_s,
            "observed_samples": len(self.samples),
            "sample_errors": self.sample_errors,
        }


def service_curve_utilization(states, now, bandwidth_bytes_s, safety=0.9):
    """Maximum EDF demand/service ratio for the currently missing bytes."""
    if bandwidth_bytes_s <= 0 or not 0 < safety <= 1:
        raise ValueError("bandwidth and safety must be positive")
    cumulative = 0.0
    peak = 0.0
    for state in sorted(
        states.values(), key=lambda item: (item["deadline"], item["session_id"])
    ):
        cumulative += max(
            0.0, state["total_bytes"] - state["applied_target_bytes"]
        )
        capacity = safety * bandwidth_bytes_s * max(
            0.0, state["deadline"] - now
        )
        if cumulative > 0:
            peak = max(peak, math.inf if capacity <= 0 else cumulative / capacity)
    return peak


def plan_service_curve_targets(
    states,
    now,
    bandwidth_bytes_s,
    safety=0.9,
    eviction_need_bytes=math.inf,
):
    """Allocate the largest EDF-feasible missing-byte curve, far deadlines first.

    ``committed_missing_bytes`` protects already accepted Evict work ahead of
    new admissions. ``max_missing_bytes`` is the most a wait may currently
    have outside HBM. ``eviction_need_bytes`` caps only new missing work; the
    PCIe curve may further reduce it. Reloaded bytes remain ineligible until a
    later conditional-deadline epoch explicitly reopens them.
    """
    if bandwidth_bytes_s <= 0 or not 0 < safety <= 1:
        raise ValueError("bandwidth and safety must be positive")
    if not states:
        return {}, {}, {}
    deadlines = sorted({state["deadline"] for state in states.values()})
    capacity = {
        deadline: safety
        * bandwidth_bytes_s
        * max(0.0, deadline - now)
        for deadline in deadlines
    }
    demand = {deadline: 0.0 for deadline in deadlines}
    allocated = {session_id: 0.0 for session_id in states}
    ordered = sorted(
        states.values(),
        key=lambda item: (item["deadline"], item["session_id"]),
        reverse=True,
    )
    def allocate(state, offered):
        affected = [
            deadline for deadline in deadlines
            if deadline >= state["deadline"]
        ]
        headroom = min(capacity[d] - demand[d] for d in affected)
        admitted = min(
            max(0.0, offered),
            max(0.0, headroom),
            state["total_bytes"] - allocated[state["session_id"]],
        )
        allocated[state["session_id"]] += admitted
        for deadline in affected:
            demand[deadline] += admitted

    for state in ordered:
        allocate(state, state["committed_missing_bytes"])
    remaining_need = max(0.0, eviction_need_bytes)
    for state in ordered:
        if remaining_need <= 0:
            break
        before = allocated[state["session_id"]]
        allocate(
            state,
            min(
                remaining_need,
                state["max_missing_bytes"]
                - state["committed_missing_bytes"],
            ),
        )
        remaining_need -= allocated[state["session_id"]] - before
    targets = {
        session_id: state["total_bytes"] - allocated[session_id]
        for session_id, state in states.items()
    }
    return targets, demand, capacity


class ServiceCurveEDFPolicy:
    """Admit Evict bytes only below a measured cumulative PCIe service curve."""

    def __init__(
        self,
        base,
        stop,
        mean,
        sigma,
        interval,
        quantile,
        bytes_per_token,
        shared_gbps,
        safety=0.9,
        observe_bandwidth=True,
        bandwidth_poll_s=0.5,
        bandwidth_window=5,
        observe_hbm=True,
        hbm_poll_s=0.1,
        reserve_low=0.05,
        reserve_high=0.30,
        queued_allocation_tokens=1024,
        reload_chunk_gb=0.5,
        reload_cooldown_s=0.2,
        max_conditional_epochs=2,
        prefetch_lead_s=1.0,
        **kwargs,
    ):
        if not 0 < safety <= 1:
            raise ValueError("service-curve safety must be in (0, 1]")
        if not 0 <= reserve_low < reserve_high < 1:
            raise ValueError("HBM reserves must satisfy 0 <= low < high < 1")
        if (
            queued_allocation_tokens < 0
            or reload_chunk_gb <= 0
            or reload_cooldown_s <= 0
            or max_conditional_epochs < 0
            or prefetch_lead_s < 0
        ):
            raise ValueError("allocation prediction and Reload chunk are invalid")
        self.base = base
        self.stop = stop
        self.interval = interval
        self.bytes_per_token = bytes_per_token
        self.shared_bytes_s = shared_gbps * GB
        self.safety = safety
        self.mean = mean
        self.sigma = sigma
        self.quantile = quantile
        self.reserve_low = reserve_low
        self.reserve_high = reserve_high
        self.queued_allocation_tokens = queued_allocation_tokens
        self.reload_chunk_bytes = reload_chunk_gb * GB
        self.reload_cooldown_s = reload_cooldown_s
        self.max_conditional_epochs = max_conditional_epochs
        self.prefetch_lead_s = prefetch_lead_s
        self.deadline_offset = lognormal_quantile(mean, sigma, quantile)
        self.rate = ObservedServiceRate(
            base,
            self.shared_bytes_s,
            poll_s=bandwidth_poll_s,
            window=bandwidth_window,
        ) if observe_bandwidth else None
        self.hbm = ObservedHBMPressure(base, poll_s=hbm_poll_s) if observe_hbm else None
        self.last_rebalance = float("-inf")
        self.last_reload_update = float("-inf")
        self.reserve_active = False
        self.lock = threading.Lock()
        self.decision_lock = threading.Lock()
        self.waiting = {}
        self.stats = {
            "calls": 0,
            "rebalances": 0,
            "target_updates": 0,
            "reload_updates": 0,
            "evict_updates": 0,
            "full_evict_updates": 0,
            "planned_reload_bytes": 0,
            "planned_evict_bytes": 0,
            "curve_limited_bytes": 0,
            "need_limited_bytes": 0,
            "peak_service_curve_utilization": 0.0,
            "waiting_samples": 0,
            "max_waiting": 0,
            "notify_rejections": 0,
            "hbm_need_events": 0,
            "hbm_need_bytes": 0,
            "hbm_blocked_reload_bytes": 0,
            "no_need_rebalances": 0,
            "deadline_epoch_rolls": 0,
            "predicted_allocation_peak_tokens": 0,
            "reserve_activations": 0,
            "opportunistic_reload_bytes": 0,
        }

    def _service_rate(self, now):
        return self.rate.update(now) if self.rate else self.shared_bytes_s

    def _hbm_budget(self, now):
        if self.hbm is None:
            return math.inf, math.inf, None
        sample = self.hbm.update(now)
        if sample is None:
            # Missing destination telemetry must never authorize new eviction.
            return 0.0, math.inf, None
        capacity = max(0.0, sample["capacity_tokens"])
        available = max(0.0, sample["available_tokens"])
        pending = max(0.0, sample["pending_prealloc_usage"]) * capacity
        predicted = pending + max(0.0, sample["queued_requests"]) * (
            self.queued_allocation_tokens
        )
        low = self.reserve_low * capacity
        high = self.reserve_high * capacity
        effective_available = available - predicted
        if not self.reserve_active and effective_available < low:
            self.reserve_active = True
            with self.lock:
                self.stats["reserve_activations"] += 1
        elif self.reserve_active and effective_available >= high:
            self.reserve_active = False
        need_tokens = (
            max(0.0, high - effective_available)
            if self.reserve_active else 0.0
        )
        # Idle predictive Reload may intentionally consume the high-to-low
        # band. The low watermark and real pending allocation remain protected.
        headroom_tokens = max(0.0, effective_available - low)
        with self.lock:
            self.stats["predicted_allocation_peak_tokens"] = max(
                self.stats["predicted_allocation_peak_tokens"], int(predicted)
            )
            if need_tokens > 0:
                self.stats["hbm_need_events"] += 1
                self.stats["hbm_need_bytes"] += int(
                    need_tokens * self.bytes_per_token
                )
            else:
                self.stats["no_need_rebalances"] += 1
        return (
            need_tokens * self.bytes_per_token,
            headroom_tokens * self.bytes_per_token,
            sample,
        )

    def begin(self, session_id, ctx_tokens):
        now = time.monotonic()
        total = max(1, ctx_tokens * self.bytes_per_token)
        with self.lock:
            self.stats["calls"] += 1
            self.waiting[session_id] = {
                "session_id": session_id,
                "wait_started": now,
                "conditional_epochs": 0,
                "deadline": now + self.deadline_offset,
                "total_bytes": total,
                # The curve is only a safety ceiling. HBM pressure separately
                # decides how much of this eligible range is actually evicted.
                "committed_missing_bytes": 0.0,
                "max_missing_bytes": float(total),
                "applied_target_bytes": float(total),
            }
            self.stats["max_waiting"] = max(
                self.stats["max_waiting"], len(self.waiting)
            )
        self.rebalance(force=True)

    def cancel(self, session_id):
        with self.lock:
            self.waiting.pop(session_id, None)

    def finish(self, session_id):
        with self.decision_lock:
            with self.lock:
                state = self.waiting.pop(session_id, None)
            if state is not None:
                notify_tool_probability(self.base, session_id, 1.0)

    def rebalance(self, *, force=False):
        if not self.decision_lock.acquire(blocking=False):
            return
        try:
            now = time.monotonic()
            if not force and now - self.last_rebalance < self.interval:
                return
            self.last_rebalance = now
            eviction_need, hbm_headroom, hbm_sample = self._hbm_budget(now)
            with self.lock:
                if eviction_need > 0:
                    for live in self.waiting.values():
                        if (
                            live["applied_target_bytes"]
                            >= live["total_bytes"] - 1.0
                            and now >= live["deadline"]
                            and live["conditional_epochs"]
                            < self.max_conditional_epochs
                        ):
                            elapsed = max(0.0, now - live["wait_started"])
                            live["deadline"] = now + conditional_quantile_remaining(
                                elapsed,
                                self.mean,
                                self.sigma,
                                self.quantile,
                            )
                            live["max_missing_bytes"] = live["total_bytes"]
                            live["conditional_epochs"] += 1
                            self.stats["deadline_epoch_rolls"] += 1
                snapshot = {
                    session_id: dict(state)
                    for session_id, state in self.waiting.items()
                }
            service_rate = self._service_rate(now)
            curve_now = now + self.prefetch_lead_s
            utilization = service_curve_utilization(
                snapshot, curve_now, service_rate, self.safety
            )
            targets, _, _ = plan_service_curve_targets(
                snapshot,
                curve_now,
                service_rate,
                self.safety,
                eviction_need_bytes=eviction_need,
            )
            reload_wanted = sum(
                max(
                    0.0,
                    target - snapshot[session_id]["applied_target_bytes"],
                )
                for session_id, target in targets.items()
            )
            if hbm_headroom < reload_wanted:
                targets, _, _ = plan_service_curve_targets(
                    snapshot,
                    curve_now,
                    service_rate,
                    self.safety,
                    eviction_need_bytes=(
                        eviction_need + reload_wanted - hbm_headroom
                    ),
                )
            opportunistic_added = 0.0
            if (
                hbm_sample is not None
                and not self.reserve_active
                and eviction_need <= 0
                and hbm_sample["pending_prealloc_usage"] <= 0
                and hbm_sample["queued_requests"] <= 0
            ):
                mandatory_reload = sum(
                    max(
                        0.0,
                        target
                        - snapshot[session_id]["applied_target_bytes"],
                    )
                    for session_id, target in targets.items()
                )
                idle_budget = max(0.0, hbm_headroom - mandatory_reload)
                for state in sorted(
                    snapshot.values(),
                    key=lambda item: (item["deadline"], item["session_id"]),
                ):
                    session_id = state["session_id"]
                    before = targets[session_id]
                    desired = min(
                        state["total_bytes"],
                        max(before, state["applied_target_bytes"])
                        + idle_budget,
                    )
                    added = max(0.0, desired - before)
                    targets[session_id] = desired
                    idle_budget -= added
                    opportunistic_added += added
                    if idle_budget <= 1.0:
                        break
            with self.lock:
                self.stats["rebalances"] += 1
                self.stats["waiting_samples"] += len(snapshot)
                self.stats["peak_service_curve_utilization"] = max(
                    self.stats["peak_service_curve_utilization"], utilization
                )
                eligible = sum(
                    max(
                        0.0,
                        state["max_missing_bytes"]
                        - state["committed_missing_bytes"],
                    )
                    for state in snapshot.values()
                )
                offered = min(eligible, eviction_need)
                admitted = sum(
                    max(
                        0.0,
                        state["total_bytes"]
                        - targets[session_id]
                        - state["committed_missing_bytes"],
                    )
                    for session_id, state in snapshot.items()
                )
                self.stats["curve_limited_bytes"] += int(
                    max(0.0, offered - admitted)
                )
                self.stats["need_limited_bytes"] += int(
                    max(0.0, eligible - offered)
                )
                self.stats["opportunistic_reload_bytes"] += int(
                    opportunistic_added
                )

            decreases = []
            increases = []
            for session_id, target in targets.items():
                state = snapshot[session_id]
                current = state["applied_target_bytes"]
                if abs(target - current) <= 1.0:
                    continue
                item = (state["deadline"], session_id, target, current)
                (decreases if target < current else increases).append(item)

            for _, session_id, target, current in sorted(
                decreases, reverse=True
            ):
                with self.lock:
                    if session_id not in self.waiting:
                        continue
                state = snapshot[session_id]
                accepted = notify_tool_probability(
                    self.base,
                    session_id,
                    min(1.0, max(0.0, target / state["total_bytes"])),
                )
                with self.lock:
                    live = self.waiting.get(session_id)
                    if not accepted:
                        self.stats["notify_rejections"] += 1
                        continue
                    if live is None:
                        continue
                    live["applied_target_bytes"] = target
                    missing = max(
                        0.0, live["total_bytes"] - target
                    )
                    live["committed_missing_bytes"] = missing
                    self.stats["target_updates"] += 1
                    self.stats["evict_updates"] += 1
                    self.stats["planned_evict_bytes"] += int(current - target)
                    self.stats["full_evict_updates"] += int(target <= 1.0)
                    hbm_headroom += current - target
                    if self.hbm is not None:
                        self.hbm.adjust_available_tokens(
                            (current - target) / self.bytes_per_token
                        )

            for _, session_id, desired, current in sorted(increases):
                if now - self.last_reload_update < self.reload_cooldown_s:
                    break
                target = min(
                    desired,
                    current + self.reload_chunk_bytes,
                    current + hbm_headroom,
                )
                if target <= current + 1.0:
                    with self.lock:
                        self.stats["hbm_blocked_reload_bytes"] += int(
                            desired - current
                        )
                    continue
                with self.lock:
                    if session_id not in self.waiting:
                        continue
                state = snapshot[session_id]
                accepted = notify_tool_probability(
                    self.base,
                    session_id,
                    min(1.0, max(0.0, target / state["total_bytes"])),
                )
                with self.lock:
                    live = self.waiting.get(session_id)
                    if not accepted:
                        self.stats["notify_rejections"] += 1
                        continue
                    if live is None:
                        continue
                    live["applied_target_bytes"] = target
                    missing = max(0.0, live["total_bytes"] - target)
                    live["committed_missing_bytes"] = missing
                    # Reloaded bytes cannot be re-Evicted in this deadline
                    # epoch. A later conditional epoch can reopen eligibility.
                    live["max_missing_bytes"] = missing
                    self.stats["target_updates"] += 1
                    self.stats["reload_updates"] += 1
                    self.stats["planned_reload_bytes"] += int(target - current)
                    hbm_headroom -= target - current
                    self.last_reload_update = now
                    if self.hbm is not None:
                        self.hbm.adjust_available_tokens(
                            -(target - current) / self.bytes_per_token
                        )
        finally:
            self.decision_lock.release()

    def wait(self, session_id, duration, ctx_tokens):
        self.begin(session_id, ctx_tokens)
        started = time.monotonic()
        while True:
            remaining = duration - (time.monotonic() - started)
            if remaining <= 0:
                self.finish(session_id)
                return False
            if self.stop.wait(min(self.interval, remaining)):
                self.cancel(session_id)
                return True
            self.rebalance()

    def snapshot_stats(self):
        with self.lock:
            result = dict(self.stats)
            result.update(
                version="9.1-demand-gated-service-curve",
                service_curve_safety=self.safety,
                hbm_reserve_high=self.reserve_high,
                hbm_reserve_low=self.reserve_low,
                reload_chunk_bytes=self.reload_chunk_bytes,
                reload_cooldown_s=self.reload_cooldown_s,
                max_conditional_epochs=self.max_conditional_epochs,
                prefetch_lead_s=self.prefetch_lead_s,
                waiting_at_end=len(self.waiting),
            )
            result["service_rate"] = (
                self.rate.snapshot() if self.rate else {
                    "configured_ceiling_bytes_s": self.shared_bytes_s,
                    "current_bytes_s": self.shared_bytes_s,
                    "observed_samples": 0,
                    "sample_errors": 0,
                }
            )
            result["hbm_pressure"] = (
                self.hbm.snapshot() if self.hbm else None
            )
        return result


def select_feasible_deadline_set(candidates, now, bandwidth_bytes_s):
    """Keep a largest feasible EDF subset, dropping the largest rescue cost.

    Each candidate is ``(session_id, absolute_deadline, remaining_bytes)``.
    The prefix feasibility test models one shared preemptive Reload link.  When
    a prefix is infeasible, removing its largest remaining job is a simple
    maximum-completion heuristic; it is deliberately used only under overload.
    """
    if bandwidth_bytes_s <= 0:
        raise ValueError("bandwidth_bytes_s must be positive")
    accepted = []
    sacrificed = []
    cumulative = 0.0
    for session_id, deadline, remaining_bytes in sorted(
        candidates, key=lambda item: (item[1], item[0])
    ):
        if remaining_bytes < 0:
            raise ValueError("remaining_bytes must be non-negative")
        accepted.append((session_id, deadline, remaining_bytes))
        cumulative += remaining_bytes
        capacity = bandwidth_bytes_s * max(0.0, deadline - now)
        while accepted and cumulative > capacity + 1e-6:
            victim = max(accepted, key=lambda item: (item[2], item[0]))
            accepted.remove(victim)
            cumulative -= victim[2]
            sacrificed.append(victim[0])
    return {item[0] for item in accepted}, sacrificed


def choose_bounded_victims(
    *, retained_current, eligible_new, remaining_by_id, max_victims
):
    """Bound victim count while preferring stable and high-benefit choices."""
    if max_victims < 0:
        raise ValueError("max_victims must be non-negative")
    candidates = set(retained_current) | set(eligible_new)
    ordered = sorted(
        candidates,
        key=lambda session_id: (
            session_id in retained_current,
            remaining_by_id.get(session_id, 0.0),
            session_id,
        ),
        reverse=True,
    )
    return set(ordered[:max_victims])


class SelectiveSheddingDeadlinePolicy:
    """Use deadline residency normally and shed only infeasible tool waits."""

    def __init__(self, base, stop, mean, sigma, bytes_per_token,
                 bandwidth_gbps, capacity_gbps, interval, quantile,
                 max_sacrifice_fraction, overload_cycles, recover_cycles):
        self.base = base
        self.stop = stop
        self.bytes_per_token = bytes_per_token
        self.planning_bandwidth_bytes_s = bandwidth_gbps * GB * 0.9
        self.capacity_bytes_s = capacity_gbps * GB * 0.9
        self.interval = interval
        self.deadline_offset = lognormal_quantile(mean, sigma, quantile)
        self.max_sacrifice_fraction = max_sacrifice_fraction
        self.overload_cycles = overload_cycles
        self.recover_cycles = recover_cycles
        if not 0.0 <= max_sacrifice_fraction < 1.0:
            raise ValueError("max_sacrifice_fraction must be in [0, 1)")
        if overload_cycles <= 0 or recover_cycles <= 0:
            raise ValueError("cycle thresholds must be positive")
        self.lock = threading.Lock()
        self.decision_lock = threading.Lock()
        self.last_rebalance_time = 0.0
        self.waiting = {}
        self.stats = {
            "calls": 0,
            "rebalances": 0,
            "overload_rebalances": 0,
            "sacrificed_calls": 0,
            "sacrifice_transitions": 0,
            "readmit_transitions": 0,
            "unresolved_overload_rebalances": 0,
            "protected_finishes": 0,
            "sacrificed_finishes": 0,
            "target_updates": 0,
            "waiting_samples": 0,
            "protected_samples": 0,
            "max_waiting": 0,
            "max_sacrificed_waiting": 0,
            "planning_bandwidth_gbps": bandwidth_gbps,
            "feasibility_capacity_gbps": capacity_gbps,
            "max_sacrifice_fraction": max_sacrifice_fraction,
            "overload_cycles": overload_cycles,
            "recover_cycles": recover_cycles,
        }

    def begin(self, session_id, ctx_tokens):
        notify_tool_probability(self.base, session_id, 0.0)
        now = time.monotonic()
        with self.lock:
            self.waiting[session_id] = {
                "started": now,
                "deadline": now + self.deadline_offset,
                "total_bytes": max(1, ctx_tokens * self.bytes_per_token),
                "target_fraction": 0.0,
                "sacrificed": False,
                "ever_sacrificed": False,
                "victim_strikes": 0,
                "recover_strikes": 0,
            }
            self.stats["calls"] += 1
        self.rebalance(force=True)

    def cancel(self, session_id):
        with self.lock:
            self.waiting.pop(session_id, None)

    def finish(self, session_id):
        with self.decision_lock:
            with self.lock:
                state = self.waiting.pop(session_id, None)
                if state is not None:
                    key = (
                        "sacrificed_finishes"
                        if state["sacrificed"]
                        else "protected_finishes"
                    )
                    self.stats[key] += 1
            # A sacrificed call is allowed to miss its proactive deadline, but
            # the actual request must still become usable when the tool returns.
            notify_tool_probability(self.base, session_id, 1.0)

    def rebalance(self, *, force=False):
        if not self.decision_lock.acquire(blocking=False):
            return
        try:
            now = time.monotonic()
            if not force and now - self.last_rebalance_time < self.interval:
                return
            self.last_rebalance_time = now
            with self.lock:
                snapshot = {key: dict(value) for key, value in self.waiting.items()}

            candidates = []
            remaining_by_id = {}
            for session_id, state in snapshot.items():
                remaining = state["total_bytes"] * (1.0 - state["target_fraction"])
                remaining_by_id[session_id] = remaining
                candidates.append((session_id, state["deadline"], remaining))
            _, raw_victim_list = select_feasible_deadline_set(
                candidates, now, self.capacity_bytes_s
            )
            raw_victims = set(raw_victim_list)

            with self.lock:
                live = set(self.waiting)
                for session_id, state in self.waiting.items():
                    if session_id in raw_victims:
                        state["victim_strikes"] += 1
                        state["recover_strikes"] = 0
                    else:
                        state["victim_strikes"] = 0
                        state["recover_strikes"] += 1

                current_victims = {
                    session_id for session_id, state in self.waiting.items()
                    if state["sacrificed"]
                }
                retained_current = {
                    session_id for session_id in current_victims
                    if (
                        session_id in raw_victims
                        or self.waiting[session_id]["recover_strikes"]
                        < self.recover_cycles
                    )
                }
                eligible_new = {
                    session_id for session_id in raw_victims - current_victims
                    if self.waiting[session_id]["victim_strikes"]
                    >= self.overload_cycles
                }
                max_victims = math.floor(
                    len(live) * self.max_sacrifice_fraction
                )
                desired_victims = choose_bounded_victims(
                    retained_current=retained_current,
                    eligible_new=eligible_new,
                    remaining_by_id=remaining_by_id,
                    max_victims=max_victims,
                )
                to_sacrifice = desired_victims - current_victims
                to_readmit = current_victims - desired_victims
                for session_id in to_sacrifice:
                    state = self.waiting[session_id]
                    if not state["ever_sacrificed"]:
                        state["ever_sacrificed"] = True
                        self.stats["sacrificed_calls"] += 1
                    state["sacrificed"] = True
                for session_id in to_readmit:
                    self.waiting[session_id]["sacrificed"] = False
                if raw_victims:
                    self.stats["overload_rebalances"] += 1
                if len(raw_victims) > len(desired_victims):
                    self.stats["unresolved_overload_rebalances"] += 1
                self.stats["sacrifice_transitions"] += len(to_sacrifice)
                self.stats["readmit_transitions"] += len(to_readmit)
                self.stats["rebalances"] += 1
                protected = live - desired_victims
                protected_snapshot = {
                    session_id: dict(self.waiting[session_id])
                    for session_id in protected
                }
                sacrificed_waiting = len(desired_victims)
                self.stats["waiting_samples"] += len(live)
                self.stats["protected_samples"] += len(protected)
                self.stats["max_waiting"] = max(
                    self.stats["max_waiting"], len(live)
                )
                self.stats["max_sacrificed_waiting"] = max(
                    self.stats["max_sacrificed_waiting"], sacrificed_waiting
                )

            updates = []
            for session_id, state in protected_snapshot.items():
                slack = max(0.0, state["deadline"] - now)
                next_fraction = deadline_residency_fraction(
                    total_bytes=state["total_bytes"],
                    seconds_to_deadline=slack,
                    bandwidth_bytes_s=self.planning_bandwidth_bytes_s,
                    safety=1.0,
                )
                next_fraction = max(state["target_fraction"], next_fraction)
                if next_fraction > state["target_fraction"] + 1e-6:
                    updates.append((session_id, next_fraction))

            for session_id, target_fraction in sorted(updates):
                with self.lock:
                    state = self.waiting.get(session_id)
                    if state is None or state["sacrificed"]:
                        continue
                notify_tool_probability(self.base, session_id, target_fraction)
                with self.lock:
                    state = self.waiting.get(session_id)
                    if state is not None and not state["sacrificed"]:
                        state["target_fraction"] = max(
                            state["target_fraction"], target_fraction
                        )
                        self.stats["target_updates"] += 1
        finally:
            self.decision_lock.release()

    def wait(self, session_id, duration, ctx_tokens):
        self.begin(session_id, ctx_tokens)
        started = time.monotonic()
        while True:
            remaining = duration - (time.monotonic() - started)
            if remaining <= 0:
                self.finish(session_id)
                return False
            if self.stop.wait(min(self.interval, remaining)):
                self.cancel(session_id)
                return True
            self.rebalance()

    def snapshot_stats(self):
        with self.lock:
            result = dict(self.stats)
            if result["waiting_samples"]:
                result["mean_protected_fraction"] = round(
                    result["protected_samples"] / result["waiting_samples"], 4
                )
            else:
                result["mean_protected_fraction"] = 0.0
            return result


class PressureEvictionDeadlinePolicy(SelectiveSheddingDeadlinePolicy):
    """Keep 1.0 intact and add a conservative emergency pressure valve.

    A wait can be evicted only when its conditional next-return deadline is
    outside a small guard.  A pressure re-admission also starts a cooldown so
    pressure oscillation cannot immediately evict the same KV.
    """

    def __init__(self, base, stop, mean, sigma, bytes_per_token,
                 bandwidth_gbps, interval, quantile, pressure_high,
                 pressure_low, pressure_poll_s, max_sacrifice_fraction,
                 max_sacrifice_count, recover_cycles, deadline_guard_s,
                 evict_cooldown_s):
        super().__init__(
            base=base,
            stop=stop,
            mean=mean,
            sigma=sigma,
            bytes_per_token=bytes_per_token,
            bandwidth_gbps=bandwidth_gbps,
            capacity_gbps=bandwidth_gbps,
            interval=interval,
            quantile=quantile,
            max_sacrifice_fraction=max_sacrifice_fraction,
            overload_cycles=1,
            recover_cycles=recover_cycles,
        )
        if not 0.0 <= pressure_low < pressure_high <= 1.0:
            raise ValueError("pressure watermarks must satisfy 0 <= low < high <= 1")
        if pressure_poll_s <= 0:
            raise ValueError("pressure_poll_s must be positive")
        if deadline_guard_s < 0 or evict_cooldown_s < 0:
            raise ValueError("pressure guard and cooldown must be non-negative")
        if max_sacrifice_count <= 0:
            raise ValueError("max_sacrifice_count must be positive")
        self.pressure_high = pressure_high
        self.pressure_low = pressure_low
        self.pressure_poll_s = pressure_poll_s
        self.mean = mean
        self.sigma = sigma
        self.quantile = quantile
        self.deadline_guard_s = deadline_guard_s
        self.evict_cooldown_s = evict_cooldown_s
        self.max_sacrifice_count = max_sacrifice_count
        self.pressure_mode = False
        self.last_pressure_poll = 0.0
        self.last_token_usage = 0.0
        self.pressure_recover_strikes = 0
        self.stats.update({
            "pressure_high": pressure_high,
            "pressure_low": pressure_low,
            "pressure_poll_s": pressure_poll_s,
            "deadline_guard_s": deadline_guard_s,
            "evict_cooldown_s": evict_cooldown_s,
            "max_sacrifice_count": max_sacrifice_count,
            "pressure_polls": 0,
            "pressure_scrape_errors": 0,
            "pressure_entries": 0,
            "pressure_exits": 0,
            "pressure_samples_high": 0,
            "peak_token_usage": 0.0,
            "active_evict_transitions": 0,
            "active_reclaims_planned_bytes": 0,
            "deadline_guard_skips": 0,
            "cooldown_skips": 0,
            "pressure_polls_without_safe_victim": 0,
        })

    def rebalance(self, *, force=False):
        if not self.decision_lock.acquire(blocking=False):
            return
        try:
            now = time.monotonic()
            if not force and now - self.last_rebalance_time < self.interval:
                return
            self.last_rebalance_time = now

            polled = False
            if now - self.last_pressure_poll >= self.pressure_poll_s:
                self.last_pressure_poll = now
                polled = True
                try:
                    usage = scrape_token_usage(self.base)
                except Exception:
                    usage = None
                    self.stats["pressure_scrape_errors"] += 1
                if usage is not None:
                    self.last_token_usage = usage
                    self.stats["pressure_polls"] += 1
                    self.stats["peak_token_usage"] = max(
                        self.stats["peak_token_usage"], usage
                    )
                    if usage >= self.pressure_high:
                        self.stats["pressure_samples_high"] += 1
                        self.pressure_recover_strikes = 0
                        if not self.pressure_mode:
                            self.pressure_mode = True
                            self.stats["pressure_entries"] += 1
                    elif usage <= self.pressure_low:
                        self.pressure_recover_strikes += 1
                        if (
                            self.pressure_mode
                            and self.pressure_recover_strikes >= self.recover_cycles
                        ):
                            self.pressure_mode = False
                            self.stats["pressure_exits"] += 1
                    else:
                        self.pressure_recover_strikes = 0

            with self.lock:
                live = set(self.waiting)
                current_victims = {
                    session_id for session_id, state in self.waiting.items()
                    if state["sacrificed"]
                }
                max_victims = math.floor(
                    len(live) * self.max_sacrifice_fraction
                )
                max_victims = min(max_victims, self.max_sacrifice_count)
                to_evict = None
                to_readmit = None
                if (
                    polled
                    and self.pressure_mode
                    and self.last_token_usage >= self.pressure_high
                    and len(current_victims) < max_victims
                ):
                    choices = []
                    for session_id, state in self.waiting.items():
                        if state["sacrificed"] or state["target_fraction"] <= 0:
                            continue
                        started = state.get(
                            "started", state["deadline"] - self.deadline_offset
                        )
                        elapsed_s = max(0.0, now - started)
                        slack_s = conditional_quantile_remaining(
                            elapsed_s,
                            self.mean,
                            self.sigma,
                            self.quantile,
                        )
                        if slack_s <= self.deadline_guard_s:
                            self.stats["deadline_guard_skips"] += 1
                            continue
                        if now < state.get("evict_cooldown_until", 0.0):
                            self.stats["cooldown_skips"] += 1
                            continue
                        reclaimable = (
                            state["target_fraction"] * state["total_bytes"]
                        )
                        choices.append((reclaimable, slack_s, session_id))
                    if choices:
                        reclaimable, _, to_evict = max(choices)
                        state = self.waiting[to_evict]
                        state["sacrificed"] = True
                        if not state["ever_sacrificed"]:
                            state["ever_sacrificed"] = True
                            self.stats["sacrificed_calls"] += 1
                        self.stats["sacrifice_transitions"] += 1
                        self.stats["active_evict_transitions"] += 1
                        self.stats["active_reclaims_planned_bytes"] += int(
                            reclaimable
                        )
                        current_victims.add(to_evict)
                    else:
                        self.stats["pressure_polls_without_safe_victim"] += 1
                elif polled and not self.pressure_mode and current_victims:
                    # Re-admit one request per pressure poll, earliest deadline first.
                    to_readmit = min(
                        current_victims,
                        key=lambda session_id: self.waiting[session_id]["deadline"],
                    )
                    self.waiting[to_readmit]["sacrificed"] = False
                    self.waiting[to_readmit]["evict_cooldown_until"] = (
                        now + self.evict_cooldown_s
                    )
                    self.stats["readmit_transitions"] += 1
                    current_victims.remove(to_readmit)

                protected = live - current_victims
                protected_snapshot = {
                    session_id: dict(self.waiting[session_id])
                    for session_id in protected
                }
                self.stats["rebalances"] += 1
                self.stats["waiting_samples"] += len(live)
                self.stats["protected_samples"] += len(protected)
                self.stats["max_waiting"] = max(
                    self.stats["max_waiting"], len(live)
                )
                self.stats["max_sacrificed_waiting"] = max(
                    self.stats["max_sacrificed_waiting"], len(current_victims)
                )

            # A new pressure victim is actively demoted instead of merely
            # losing future Reload service.
            if to_evict is not None:
                notify_tool_probability(self.base, to_evict, 0.0)
                with self.lock:
                    state = self.waiting.get(to_evict)
                    if state is not None and state["sacrificed"]:
                        state["target_fraction"] = 0.0

            updates = []
            for session_id, state in protected_snapshot.items():
                slack = max(0.0, state["deadline"] - now)
                next_fraction = deadline_residency_fraction(
                    total_bytes=state["total_bytes"],
                    seconds_to_deadline=slack,
                    bandwidth_bytes_s=self.planning_bandwidth_bytes_s,
                    safety=1.0,
                )
                next_fraction = max(state["target_fraction"], next_fraction)
                if next_fraction > state["target_fraction"] + 1e-6:
                    updates.append((session_id, next_fraction))

            for session_id, target_fraction in sorted(updates):
                with self.lock:
                    state = self.waiting.get(session_id)
                    if state is None or state["sacrificed"]:
                        continue
                notify_tool_probability(self.base, session_id, target_fraction)
                with self.lock:
                    state = self.waiting.get(session_id)
                    if state is not None and not state["sacrificed"]:
                        state["target_fraction"] = max(
                            state["target_fraction"], target_fraction
                        )
                        self.stats["target_updates"] += 1
        finally:
            self.decision_lock.release()


class GoodputAdmissionPolicy:
    """Protect only the waiting sessions with the best success gain per HBM byte.

    This is deliberately an experiment-harness policy: it uses only the known
    latency distribution, elapsed wait, context size, and configured bandwidth.
    It never reads the sampled tool completion time.
    """

    def __init__(self, base, stop, mean, sigma, bytes_per_token, budget_gb,
                 bandwidth_gbps, interval, quantile):
        self.base = base
        self.stop = stop
        self.mean = mean
        self.sigma = sigma
        self.bytes_per_token = bytes_per_token
        self.budget_bytes = budget_gb * GB
        self.bandwidth_bytes_s = bandwidth_gbps * GB * 0.9
        self.interval = interval
        self.quantile = quantile
        self.mu = math.log(mean) - sigma * sigma / 2
        self.lock = threading.Lock()
        self.decision_lock = threading.Lock()
        self.waiting = {}
        self.admitted = set()
        self.stats = {
            "rebalances": 0,
            "admit_transitions": 0,
            "reject_transitions": 0,
            "protected_samples": 0,
            "waiting_samples": 0,
            "max_protected": 0,
        }

    def _conditional_survival(self, elapsed, future):
        survived = max(1e-12, 1.0 - tool_return_cdf(elapsed, self.mean, self.sigma))
        return min(
            1.0,
            max(0.0, (1.0 - tool_return_cdf(future, self.mean, self.sigma)) / survived),
        )

    def _conditional_quantile_remaining(self, elapsed):
        current_cdf = tool_return_cdf(elapsed, self.mean, self.sigma)
        target_cdf = current_cdf + (1.0 - current_cdf) * self.quantile
        z = statistics.NormalDist().inv_cdf(min(1 - 1e-9, target_cdf))
        return max(0.0, math.exp(self.mu + self.sigma * z) - elapsed)

    def begin(self, session_id, ctx_tokens):
        notify_tool_probability(self.base, session_id, 0.0)
        with self.lock:
            self.waiting[session_id] = {
                "started": time.monotonic(),
                "ctx_tokens": ctx_tokens,
            }
        self.rebalance()

    def cancel(self, session_id):
        with self.lock:
            self.waiting.pop(session_id, None)
            self.admitted.discard(session_id)

    def finish(self, session_id):
        with self.decision_lock:
            with self.lock:
                self.waiting.pop(session_id, None)
                self.admitted.discard(session_id)
            # Every returned request must be usable. Rejected sessions pay the
            # demand-Reload latency here, which is exactly the intended sacrifice.
            notify_tool_probability(self.base, session_id, 1.0)

    def rebalance(self):
        if not self.decision_lock.acquire(blocking=False):
            return
        try:
            now = time.monotonic()
            with self.lock:
                snapshot = dict(self.waiting)
                previously_admitted = set(self.admitted)

            candidates = []
            for session_id, state in snapshot.items():
                elapsed = max(0.0, now - state["started"])
                size = max(1, state["ctx_tokens"] * self.bytes_per_token)
                reload_s = size / self.bandwidth_bytes_s
                success = self._conditional_survival(elapsed, elapsed + reload_s)
                deadline = self._conditional_quantile_remaining(elapsed)
                # Marginal expected goodput per protected HBM byte. A small
                # hysteresis bonus prevents the selected set from oscillating.
                density = success / size
                if session_id in previously_admitted:
                    density *= 1.15
                candidates.append((density, deadline, size, session_id, success))

            selected = set()
            used = 0
            for _, _, size, session_id, _ in sorted(
                candidates, key=lambda item: (-item[0], item[1], item[3])
            ):
                if used + size <= self.budget_bytes:
                    selected.add(session_id)
                    used += size

            to_admit = sorted(selected - previously_admitted)
            to_reject = sorted(previously_admitted - selected)
            for session_id in to_reject:
                notify_tool_probability(self.base, session_id, 0.0)
            for session_id in to_admit:
                notify_tool_probability(self.base, session_id, 1.0)

            with self.lock:
                # Drop sessions that completed while HTTP updates were issued.
                live = set(self.waiting)
                self.admitted = selected & live
                self.stats["rebalances"] += 1
                self.stats["admit_transitions"] += len(to_admit)
                self.stats["reject_transitions"] += len(to_reject)
                self.stats["protected_samples"] += len(self.admitted)
                self.stats["waiting_samples"] += len(live)
                self.stats["max_protected"] = max(
                    self.stats["max_protected"], len(self.admitted)
                )
        finally:
            self.decision_lock.release()

    def wait(self, session_id, duration, ctx_tokens):
        self.begin(session_id, ctx_tokens)
        started = time.monotonic()
        while True:
            remaining = duration - (time.monotonic() - started)
            if remaining <= 0:
                self.finish(session_id)
                return False
            if self.stop.wait(min(self.interval, remaining)):
                self.cancel(session_id)
                return True
            self.rebalance()

    def snapshot_stats(self):
        with self.lock:
            result = dict(self.stats)
            if result["waiting_samples"]:
                result["mean_protected_fraction"] = round(
                    result["protected_samples"] / result["waiting_samples"], 4
                )
            else:
                result["mean_protected_fraction"] = 0.0
            return result


def scrape(base):
    o = {}
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=8) as r:
            for ln in r.read().decode().splitlines():
                if ln.startswith("#") or "kv_offload_total" not in ln:
                    continue
                d = "c2g" if "CPU_to_GPU" in ln else (
                    "g2c" if "GPU_to_CPU" in ln else None)
                t = "bytes" if "bytes_total" in ln else (
                    "time" if "time_total" in ln else None)
                if d and t:
                    try:
                        o[f"{t}_{d}"] = float(ln.rsplit(None, 1)[1])
                    except ValueError:
                        pass
    except Exception:
        pass
    if "bytes_c2g" not in o or "time_c2g" not in o:
        try:
            counters = read_c2g_counters(base, timeout=8)
            if counters is not None:
                o["bytes_c2g"], o["time_c2g"] = counters
        except Exception:
            pass
    return o


class Tracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.n = self.peak = 0

    def enter(self):
        with self.lock:
            o = self.n
            self.n += 1
            self.peak = max(self.peak, self.n)
            return o

    def exit(self):
        with self.lock:
            self.n -= 1


def main():
    p = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    p.add_argument("--model", default=f"{home}/models/Qwen2.5-7B-Instruct")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--sessions", type=int, default=32)
    p.add_argument("--duration", type=float, default=240.0)
    p.add_argument("--gen", type=int, default=16)
    p.add_argument("--tool-mean", type=float, default=2.0)
    p.add_argument("--tool-sigma", type=float, default=0.8)
    p.add_argument("--max-ctx", type=int, default=30000)
    p.add_argument("--initial-context-tokens", type=int, default=0,
                   help="Minimum token count in each session's first prompt")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--trace-cache", default=TRACES,
        help="Cache file for sampled SWE-agent turn shapes",
    )
    p.add_argument(
        "--issues-dataset", default="princeton-nlp/SWE-bench",
        help="Hugging Face dataset ID or a local load_from_disk directory",
    )
    p.add_argument("--issues-split", default="test")
    p.add_argument("--probability-residency", action="store_true",
                   help="Actively adjust session HBM residency during tool waits")
    p.add_argument("--probability-update-interval", type=float, default=0.5,
                   help="Seconds between proactive probability updates")
    p.add_argument("--deadline-residency", action="store_true",
                   help="Prepare KV progressively for a return-time quantile")
    p.add_argument("--deadline-quantile", type=float, default=0.10,
                   help="Conditional early-return quantile used as the deadline")
    p.add_argument("--deadline-bandwidth-gbps", type=float, default=10.0,
                   help="Effective DRAM-to-HBM Reload bandwidth")
    p.add_argument("--shared-bandwidth-residency", action="store_true",
                   help="Run 1.0 with a shared Reload-bandwidth correction")
    p.add_argument("--shared-pressure-shedding", action="store_true",
                   help="Run shared-bandwidth 3.0 plus one-victim pressure shedding")
    p.add_argument("--deadline-floor", action="store_true",
                   help=("Demand-only eviction with a shared-EDF deadline "
                         "residency floor"))
    p.add_argument(
        "--service-curve-edf",
        "--adaptive-bandwidth-v4-lru",
        "--adaptive-low-pressure-v3",
        dest="service_curve_edf",
        action="store_true",
        help="Admit Evict bytes under a measured cumulative H2D service curve",
    )
    p.add_argument("--session-cache-only", action="store_true",
                   help="Same session-cache workload without deadline-floor control")
    p.add_argument("--lru", action="store_true",
                   help="Native LRU workload without session IDs or active control")
    p.add_argument("--shared-bandwidth-gbps", type=float, default=8.0,
                   help="Conservative ceiling for aggregate H2D service")
    p.add_argument("--service-curve-safety", type=float, default=0.9)
    p.add_argument("--service-curve-interval", type=float, default=0.1)
    p.add_argument("--service-bandwidth-poll-s", type=float, default=0.5)
    p.add_argument("--service-bandwidth-window", type=int, default=5)
    p.add_argument("--no-service-bandwidth-observation", action="store_true",
                   help="Use the configured service ceiling without telemetry")
    p.add_argument("--service-hbm-poll-s", type=float, default=0.1)
    p.add_argument("--service-hbm-reserve-low", type=float, default=0.05)
    p.add_argument("--service-hbm-reserve-high", type=float, default=0.30)
    p.add_argument("--service-queued-allocation-tokens", type=int, default=1024)
    p.add_argument("--service-reload-chunk-gb", type=float, default=0.5)
    p.add_argument("--service-reload-cooldown-s", type=float, default=0.2)
    p.add_argument("--service-max-conditional-epochs", type=int, default=2)
    p.add_argument("--service-prefetch-lead-s", type=float, default=1.0)
    p.add_argument("--no-service-hbm-observation", action="store_true",
                   help="Disable demand-gating HBM telemetry")
    p.add_argument("--selective-shedding", action="store_true",
                   help="Shed only deadline-infeasible tool waits")
    p.add_argument("--shedding-capacity-gbps", type=float, default=4.0,
                   help="Aggregate Reload capacity used only for feasibility")
    p.add_argument("--shedding-max-fraction", type=float, default=0.25,
                   help="Maximum fraction of waiting calls sacrificed at once")
    p.add_argument("--shedding-overload-cycles", type=int, default=3)
    p.add_argument("--shedding-recover-cycles", type=int, default=5)
    p.add_argument("--pressure-eviction", action="store_true",
                   help="Use real SGLang HBM pressure to actively evict waits")
    p.add_argument("--pressure-high", type=float, default=0.85,
                   help="Demand-to-v4 recovery watermark for EDF bandwidth load")
    p.add_argument("--pressure-low", type=float, default=0.70,
                   help="Low-pressure reporting boundary for EDF bandwidth load")
    p.add_argument("--pressure-poll-interval", type=float, default=0.20)
    p.add_argument("--pressure-evict-quantum-gb", type=float, default=1.0,
                   help=("GiB removed from the waiting-session HBM budget "
                         "per pressure poll"))
    p.add_argument("--pressure-max-fraction", type=float, default=0.25,
                   help="Maximum fraction of waiting calls actively evicted")
    p.add_argument("--pressure-max-count", type=int, default=1,
                   help="Hard cap on concurrently pressure-evicted calls")
    p.add_argument("--pressure-recover-cycles", type=int, default=3)
    p.add_argument("--pressure-deadline-guard-s", type=float, default=0.10,
                   help="Never Evict inside this conditional deadline window")
    p.add_argument("--pressure-evict-cooldown-s", type=float, default=1.0,
                   help="Do not re-Evict a pressure-reloaded wait this soon")
    p.add_argument("--goodput-admission", action="store_true",
                   help="Protect only sessions with the best expected goodput per byte")
    p.add_argument("--goodput-hbm-budget-gb", type=float, default=2.0)
    p.add_argument("--goodput-bandwidth-gbps", type=float, default=10.0)
    p.add_argument("--goodput-quantile", type=float, default=0.10)
    p.add_argument("--goodput-slo-ms", type=float, default=250.0)
    p.add_argument("--json", type=str, default=None)
    a = p.parse_args()
    if not (
        a.service_curve_edf
        or a.deadline_floor
        or a.session_cache_only
        or a.lru
    ):
        p.error(
            "choose --service-curve-edf, --deadline-floor, "
            "--session-cache-only, or --lru"
        )
    selected_policies = sum((
        a.probability_residency,
        a.deadline_residency,
        a.shared_bandwidth_residency,
        a.shared_pressure_shedding,
        a.service_curve_edf,
        a.deadline_floor,
        a.session_cache_only,
        a.lru,
        a.selective_shedding,
        a.pressure_eviction,
        a.goodput_admission,
    ))
    if selected_policies > 1:
        p.error("choose only one residency policy")
    if not 0.0 < a.deadline_quantile < 1.0:
        p.error("--deadline-quantile must be in (0, 1)")
    if a.deadline_bandwidth_gbps <= 0:
        p.error("--deadline-bandwidth-gbps must be positive")
    if a.shared_bandwidth_gbps <= 0:
        p.error("--shared-bandwidth-gbps must be positive")
    if not 0.0 < a.service_curve_safety <= 1.0:
        p.error("--service-curve-safety must be in (0, 1]")
    if a.service_curve_interval <= 0:
        p.error("--service-curve-interval must be positive")
    if a.service_bandwidth_poll_s <= 0 or a.service_bandwidth_window <= 0:
        p.error("service-bandwidth observation parameters must be positive")
    if a.service_hbm_poll_s <= 0:
        p.error("--service-hbm-poll-s must be positive")
    if not (
        0.0 <= a.service_hbm_reserve_low
        < a.service_hbm_reserve_high < 1.0
    ):
        p.error("service HBM reserves must satisfy 0 <= low < high < 1")
    if (
        a.service_queued_allocation_tokens < 0
        or a.service_reload_chunk_gb <= 0
        or a.service_reload_cooldown_s <= 0
        or a.service_max_conditional_epochs < 0
        or a.service_prefetch_lead_s < 0
    ):
        p.error("service allocation prediction and Reload chunk are invalid")
    if a.shedding_capacity_gbps <= 0:
        p.error("--shedding-capacity-gbps must be positive")
    if not 0.0 <= a.shedding_max_fraction < 1.0:
        p.error("--shedding-max-fraction must be in [0, 1)")
    if a.shedding_overload_cycles <= 0 or a.shedding_recover_cycles <= 0:
        p.error("shedding cycle thresholds must be positive")
    if not 0.0 <= a.pressure_low < a.pressure_high <= 1.0:
        p.error("pressure watermarks must satisfy 0 <= low < high <= 1")
    if a.pressure_poll_interval <= 0:
        p.error("--pressure-poll-interval must be positive")
    if a.pressure_evict_quantum_gb <= 0:
        p.error("--pressure-evict-quantum-gb must be positive")
    if not 0.0 <= a.pressure_max_fraction < 1.0:
        p.error("--pressure-max-fraction must be in [0, 1)")
    if a.pressure_recover_cycles <= 0:
        p.error("--pressure-recover-cycles must be positive")
    if a.pressure_max_count <= 0:
        p.error("--pressure-max-count must be positive")
    if a.pressure_deadline_guard_s < 0 or a.pressure_evict_cooldown_s < 0:
        p.error("pressure deadline guard and cooldown must be non-negative")

    base = f"http://127.0.0.1:{a.port}"
    comp = base + "/v1/completions"
    try:
        urllib.request.urlopen(base + "/v1/models", timeout=10).read()
    except Exception:
        sys.exit(f"model server not reachable at {base}")

    c = get_model_config(a.model)
    L = c["num_hidden_layers"]
    Hkv = c.get("num_key_value_heads", c["num_attention_heads"])
    hd = c.get("head_dim", c["hidden_size"] // c["num_attention_heads"])
    b = 2 * L * Hkv * hd * 2

    print("=" * 76)
    print("  SWE-AGENT SHAPE REPLAY -- real issue text, synthetic observations")
    print("=" * 76)
    tr_data = build_traces(a.model, out=os.path.expanduser(a.trace_cache))
    allc = sorted(x["ctx_before"] for s in tr_data for x in s["turns"])
    allo = sorted(x["obs"] for s in tr_data for x in s["turns"])
    print(f"\n  {len(tr_data)} recorded SWE-agent turn-length sequences")
    print(f"  turns/session : p50 "
          f"{sorted(len(s['turns']) for s in tr_data)[len(tr_data)//2]}, "
          f"max {max(len(s['turns']) for s in tr_data)}")
    print(f"  context       : p50 {allc[len(allc)//2]}, "
          f"p90 {allc[int(.9*len(allc))]}, max {allc[-1]} tokens")
    print(f"  observation   : p50 {allo[len(allo)//2]}, "
          f"p90 {allo[int(.9*len(allo))]} tokens")
    print(f"  asymmetry     : reload is "
          f"{allc[len(allc)//2] / max(allo[len(allo)//2], 1):.0f}x the prefill "
          f"at the median turn")
    print(f"\n  KV {b / KiB:.0f} KiB/token -> p50 restore "
          f"{allc[len(allc)//2] * b / GB:.2f} GB, "
          f"p90 {allc[int(.9*len(allc))] * b / GB:.2f} GB")
    print(f"\n  {a.sessions} independent sessions, lognormal tool latency "
          f"(mean {a.tool_mean}s)")
    print("  no barrier, no groups -- every burst is a collision")

    print("\n  pre-building prompts (avoids tokenizing inside worker threads,")
    print("  where a failure would silently kill a session) ...", flush=True)
    get_tok(a.model)
    from datasets import load_dataset, load_from_disk
    issues_path = Path(a.issues_dataset).expanduser()
    issues = (
        load_from_disk(str(issues_path))
        if issues_path.exists()
        else load_dataset(a.issues_dataset, split=a.issues_split)
    )
    prebuilt = {}
    for sid in range(a.sessions):
        tr = tr_data[sid % len(tr_data)]
        issue = issues[sid % len(issues)]
        issue_text = (
            f"Repository: {issue['repo']}\n"
            f"Issue: {issue['problem_statement']}\n"
        )
        tok = get_tok(a.model)
        issue_ids = tok(issue_text, add_special_tokens=False)["input_ids"][:1500]
        issue_text = tok.decode(issue_ids)
        target = max(64, tr["turns"][0]["ctx_before"], a.initial_context_tokens)
        base_ctx = issue_text + filler(
            f"sys{sid}", max(0, target - len(issue_ids)), a.model
        )
        obs_list = [filler(f"s{sid}t{j}", max(1, t["obs"]), a.model)
                    for j, t in enumerate(tr["turns"][:24])]
        prebuilt[sid] = (base_ctx, obs_list)
    print(f"  built {a.sessions} session prompt sets", flush=True)

    rows, lock = [], threading.Lock()
    tokenizer_lock = threading.Lock()
    errors = {}
    tk = Tracker()
    stop = threading.Event()
    rng0 = random.Random(a.seed)
    seeds = [rng0.randrange(1 << 30) for _ in range(a.sessions)]
    t0g = [0.0]
    goodput = None
    shedding = None
    pressure = None
    shared_bandwidth = None
    deadline_floor = None
    service_curve_edf = None
    if a.shared_bandwidth_residency or a.shared_pressure_shedding:
        shared_bandwidth = SharedBandwidthCoordinator(
            per_request_gbps=a.deadline_bandwidth_gbps,
            shared_gbps=a.shared_bandwidth_gbps,
            pressure_enabled=a.shared_pressure_shedding,
            pressure_high=a.pressure_high,
            pressure_low=a.pressure_low,
            pressure_poll_s=a.pressure_poll_interval,
            recover_cycles=a.pressure_recover_cycles,
            deadline_guard_s=a.pressure_deadline_guard_s,
            cooldown_s=a.pressure_evict_cooldown_s,
            mean=a.tool_mean,
            sigma=a.tool_sigma,
            quantile=a.deadline_quantile,
        )
    if a.selective_shedding:
        shedding = SelectiveSheddingDeadlinePolicy(
            base=base,
            stop=stop,
            mean=a.tool_mean,
            sigma=a.tool_sigma,
            bytes_per_token=b,
            bandwidth_gbps=a.deadline_bandwidth_gbps,
            capacity_gbps=a.shedding_capacity_gbps,
            interval=a.probability_update_interval,
            quantile=a.deadline_quantile,
            max_sacrifice_fraction=a.shedding_max_fraction,
            overload_cycles=a.shedding_overload_cycles,
            recover_cycles=a.shedding_recover_cycles,
        )
    if a.deadline_floor:
        deadline_floor = RuntimeDeadlineFloorPolicy(
            base=base,
            stop=stop,
            mean=a.tool_mean,
            sigma=a.tool_sigma,
            bytes_per_token=b,
            per_request_gbps=a.deadline_bandwidth_gbps,
            shared_gbps=a.shared_bandwidth_gbps,
            interval=a.probability_update_interval,
            quantile=a.deadline_quantile,
            pressure_high=a.pressure_high,
            pressure_low=a.pressure_low,
            pressure_poll_s=a.pressure_poll_interval,
            recover_cycles=a.pressure_recover_cycles,
            eviction_quantum_gb=a.pressure_evict_quantum_gb,
        )
    if a.service_curve_edf:
        service_curve_edf = ServiceCurveEDFPolicy(
            base=base,
            stop=stop,
            mean=a.tool_mean,
            sigma=a.tool_sigma,
            bytes_per_token=b,
            shared_gbps=a.shared_bandwidth_gbps,
            interval=a.service_curve_interval,
            quantile=a.deadline_quantile,
            safety=a.service_curve_safety,
            observe_bandwidth=not a.no_service_bandwidth_observation,
            bandwidth_poll_s=a.service_bandwidth_poll_s,
            bandwidth_window=a.service_bandwidth_window,
            observe_hbm=not a.no_service_hbm_observation,
            hbm_poll_s=a.service_hbm_poll_s,
            reserve_low=a.service_hbm_reserve_low,
            reserve_high=a.service_hbm_reserve_high,
            queued_allocation_tokens=a.service_queued_allocation_tokens,
            reload_chunk_gb=a.service_reload_chunk_gb,
            reload_cooldown_s=a.service_reload_cooldown_s,
            max_conditional_epochs=a.service_max_conditional_epochs,
            prefetch_lead_s=a.service_prefetch_lead_s,
        )
    if a.pressure_eviction:
        pressure = PressureEvictionDeadlinePolicy(
            base=base,
            stop=stop,
            mean=a.tool_mean,
            sigma=a.tool_sigma,
            bytes_per_token=b,
            bandwidth_gbps=a.deadline_bandwidth_gbps,
            interval=a.probability_update_interval,
            quantile=a.deadline_quantile,
            pressure_high=a.pressure_high,
            pressure_low=a.pressure_low,
            pressure_poll_s=a.pressure_poll_interval,
            max_sacrifice_fraction=a.pressure_max_fraction,
            max_sacrifice_count=a.pressure_max_count,
            recover_cycles=a.pressure_recover_cycles,
            deadline_guard_s=a.pressure_deadline_guard_s,
            evict_cooldown_s=a.pressure_evict_cooldown_s,
        )
    if a.goodput_admission:
        goodput = GoodputAdmissionPolicy(
            base=base,
            stop=stop,
            mean=a.tool_mean,
            sigma=a.tool_sigma,
            bytes_per_token=b,
            budget_gb=a.goodput_hbm_budget_gb,
            bandwidth_gbps=a.goodput_bandwidth_gbps,
            interval=a.probability_update_interval,
            quantile=a.goodput_quantile,
        )

    def worker(sid):
        rng = random.Random(seeds[sid])
        tr = tr_data[sid % len(tr_data)]
        turns = tr["turns"]
        # Rebuild the context to the trace's own turn-0 size, then let it grow
        # exactly as the trace does.
        base_ctx, obs_list = prebuilt[sid]
        ctx_txt = base_ctx
        session_id = f"swe-replay-{sid}"
        time.sleep(rng.uniform(0, a.tool_mean))
        has_cached_turn = False
        cached_ctx_tokens = 0
        i = 0
        while not stop.is_set():
            turn = turns[i % len(turns)]
            i += 1
            think = rng.lognormvariate(
                math.log(a.tool_mean) - a.tool_sigma ** 2 / 2, a.tool_sigma)
            wait_time = min(think, 30.0)
            if a.service_curve_edf and has_cached_turn:
                try:
                    interrupted = service_curve_edf.wait(
                        session_id, wait_time, cached_ctx_tokens
                    )
                except Exception as exc:
                    with lock:
                        key = f"service_curve_edf_{type(exc).__name__}"
                        errors[key] = errors.get(key, 0) + 1
                    break
                if interrupted:
                    break
            elif a.deadline_floor and has_cached_turn:
                try:
                    interrupted = deadline_floor.wait(
                        session_id, wait_time, cached_ctx_tokens
                    )
                except Exception as exc:
                    with lock:
                        key = f"deadline_floor_{type(exc).__name__}"
                        errors[key] = errors.get(key, 0) + 1
                    break
                if interrupted:
                    break
            elif (
                a.shared_bandwidth_residency or a.shared_pressure_shedding
            ) and has_cached_turn:
                try:
                    interrupted = wait_with_shared_bandwidth_deadline(
                        stop,
                        wait_time,
                        base,
                        session_id,
                        a.tool_mean,
                        a.tool_sigma,
                        a.probability_update_interval,
                        a.deadline_quantile,
                        cached_ctx_tokens * b,
                        shared_bandwidth,
                    )
                except Exception as exc:
                    with lock:
                        prefix = (
                            "shared_pressure"
                            if a.shared_pressure_shedding
                            else "shared_bandwidth"
                        )
                        key = f"{prefix}_{type(exc).__name__}"
                        errors[key] = errors.get(key, 0) + 1
                    break
                if interrupted:
                    break
            elif a.pressure_eviction and has_cached_turn:
                try:
                    interrupted = pressure.wait(
                        session_id, wait_time, cached_ctx_tokens
                    )
                except Exception as exc:
                    with lock:
                        key = f"pressure_{type(exc).__name__}"
                        errors[key] = errors.get(key, 0) + 1
                    break
                if interrupted:
                    break
            elif a.selective_shedding and has_cached_turn:
                try:
                    interrupted = shedding.wait(
                        session_id, wait_time, cached_ctx_tokens
                    )
                except Exception as exc:
                    with lock:
                        key = f"shedding_{type(exc).__name__}"
                        errors[key] = errors.get(key, 0) + 1
                    break
                if interrupted:
                    break
            elif a.goodput_admission and has_cached_turn:
                try:
                    interrupted = goodput.wait(
                        session_id, wait_time, cached_ctx_tokens
                    )
                except Exception as exc:
                    with lock:
                        key = f"goodput_{type(exc).__name__}"
                        errors[key] = errors.get(key, 0) + 1
                    break
                if interrupted:
                    break
            elif a.deadline_residency and has_cached_turn:
                try:
                    interrupted = wait_with_deadline_residency(
                        stop,
                        wait_time,
                        base,
                        session_id,
                        a.tool_mean,
                        a.tool_sigma,
                        a.probability_update_interval,
                        a.deadline_quantile,
                        cached_ctx_tokens * b,
                        a.deadline_bandwidth_gbps,
                    )
                except Exception as exc:
                    with lock:
                        key = f"deadline_{type(exc).__name__}"
                        errors[key] = errors.get(key, 0) + 1
                    break
                if interrupted:
                    break
            elif a.probability_residency and has_cached_turn:
                try:
                    interrupted = wait_with_probability(
                        stop,
                        wait_time,
                        base,
                        session_id,
                        a.tool_mean,
                        a.tool_sigma,
                        a.probability_update_interval,
                    )
                except Exception as exc:
                    with lock:
                        key = f"probability_{type(exc).__name__}"
                        errors[key] = errors.get(key, 0) + 1
                    break
                if interrupted:
                    break
            elif stop.wait(wait_time):
                break
            obs = obs_list[(i - 1) % len(obs_list)]
            # strict append: everything before this point is byte-identical to
            # the previous turn, so the KV for it is either resident (cache hit)
            # or must be reloaded from host -- never recomputed.
            prompt = ctx_txt + "\nObservation: " + obs + "\nAction:"
            # Count the complete prompt, rather than summing segments: BPE can
            # merge across boundaries.  The lock makes the shared tokenizer
            # deterministic and avoids the historical len(prompt)//4
            # undercount that admitted over-context requests.
            with tokenizer_lock:
                n_ctx = len(
                    get_tok(a.model)(prompt, add_special_tokens=False)["input_ids"]
                )
            if n_ctx > a.max_ctx:
                ctx_txt = base_ctx        # context window full: restart session
                continue
            conc = tk.enter()
            t_arr = time.perf_counter()
            try:
                payload = {"model": "m", "prompt": prompt,
                           "max_tokens": a.gen, "temperature": 0.0}
                if (a.probability_residency or a.deadline_residency
                        or a.shared_bandwidth_residency
                        or a.shared_pressure_shedding
                        or a.service_curve_edf
                        or a.deadline_floor
                        or a.session_cache_only
                        or a.selective_shedding
                        or a.pressure_eviction
                        or a.goodput_admission):
                    payload["session_id"] = session_id
                ttft, tot, out = stream_ttft(comp, payload)
                if ttft is not None:
                    with lock:
                        rows.append({"session": sid, "turn": i,
                                     "t": round(t_arr - t0g[0], 3),
                                     "concurrency": conc + 1,
                                     "ctx_tokens": n_ctx,
                                     "kv_gb": round(n_ctx * b / GB, 4),
                                     "obs_tokens": turn["obs"],
                                     "ttft_ms": round(ttft * 1e3, 1),
                                     "total_s": round(tot, 3)})
                    # Keep the prior prefix identical for a genuine KV-cache hit.
                    ctx_txt = prompt + out
                    has_cached_turn = True
                    cached_ctx_tokens = n_ctx
                else:
                    with lock:
                        errors["no_text"] = errors.get("no_text", 0) + 1
            except Exception as exc:
                key = type(exc).__name__
                with lock:
                    errors[key] = errors.get(key, 0) + 1
            finally:
                tk.exit()

    m0 = scrape(base)
    t0g[0] = time.perf_counter()
    th = [threading.Thread(target=worker, args=(i,), daemon=True)
          for i in range(a.sessions)]
    for t in th:
        t.start()
    print(f"\n  running {a.duration:.0f}s ...", flush=True)
    while time.perf_counter() - t0g[0] < a.duration:
        time.sleep(20)
        with lock:
            n = len(rows)
        print(f"    [{time.perf_counter() - t0g[0]:5.0f}s] {n:4d} req, "
              f"inflight {tk.n}, peak {tk.peak}", flush=True)
    stop.set()
    drain_deadline = time.perf_counter() + 90
    for t in th:
        t.join(timeout=max(0, drain_deadline - time.perf_counter()))
    undrained = sum(t.is_alive() for t in th)
    E = time.perf_counter() - t0g[0]
    m1 = scrape(base)
    print(
        f"  completed requests: {len(rows)}; errors: {sum(errors.values())} "
        f"{errors}; undrained sessions: {undrained}",
        flush=True,
    )

    warm = [r for r in rows if r["t"] > 30]
    if len(warm) < 20:
        print(f"warning: only {len(warm)} requests after warmup", flush=True)
        warm = rows
    if not warm:
        print("no completed requests", flush=True)
        if a.json:
            with open(a.json, "w") as f:
                json.dump({"args": vars(a), "elapsed_s": round(E, 2),
                           "rows": rows, "errors": errors,
                           "undrained_sessions": undrained,
                           "goodput_policy": (
                               goodput.snapshot_stats() if goodput else None
                           ),
                           "selective_shedding_policy": (
                               shedding.snapshot_stats() if shedding else None
                           ),
                           "pressure_eviction_policy": (
                               pressure.snapshot_stats() if pressure else None
                           ),
                           "deadline_floor_policy": (
                               deadline_floor.snapshot_stats()
                               if deadline_floor else None
                           ),
                           "service_curve_edf_policy": (
                               service_curve_edf.snapshot_stats()
                               if service_curve_edf else None
                           ),
                           "shared_bandwidth_policy": (
                               shared_bandwidth.snapshot_stats()
                               if shared_bandwidth else None
                           )}, f, indent=2)
        return

    print("\n" + "=" * 76)
    print("  QUEUEING UNDER SWE-AGENT-SHAPED REPLAY")
    print("=" * 76)
    ctxs = sorted(r["ctx_tokens"] for r in warm)
    kvs = sorted(r["kv_gb"] for r in warm)
    print(f"\n  {len(warm)} requests over {E:.0f}s, peak in-flight {tk.peak}")
    print(f"  context replayed : p50 {ctxs[len(ctxs)//2]}, "
          f"p90 {ctxs[int(.9*len(ctxs))]}, max {ctxs[-1]} tok")
    print(f"  KV per restore   : p50 {kvs[len(kvs)//2]:.2f} GB, "
          f"p90 {kvs[int(.9*len(kvs))]:.2f} GB, max {kvs[-1]:.2f} GB "
          f"({kvs[-1]/max(kvs[0],1e-9):.0f}x spread)")

    solo = [r["ttft_ms"] for r in warm if r["concurrency"] == 1]
    ref = statistics.median(solo) if len(solo) >= 5 else None
    tt = sorted(r["ttft_ms"] for r in warm)
    n = len(tt)
    good_count = sum(x <= a.goodput_slo_ms for x in tt)
    print(
        f"  Goodput@{a.goodput_slo_ms:.0f}ms : {good_count}/{n} "
        f"({100 * good_count / n:.1f}%)"
    )
    print(f"\n  uncontended TTFT (burst=1, n={len(solo)}) : "
          + (f"{ref:.0f} ms" if ref else "too few samples"))
    for q, lab in ((.5, "p50"), (.9, "p90"), (.99, "p99")):
        v = tt[min(n - 1, int(q * (n - 1)))]
        ratio = f"{v / ref:>6.2f}x" if ref else "    n/a"
        print(f"  {lab} TTFT : {v:>8.0f} ms   {ratio}")
    if ref:
        over = sum(1 for x in tt if x > 2 * ref)
        print(f"  requests >2x uncontended : {over} ({100 * over / n:.0f}%)")

    print(f"\n  {'burst':>6} {'n':>5} {'p50 ms':>9} {'p99 ms':>9} {'vs solo':>9}")
    h = {}
    for r in warm:
        h.setdefault(r["concurrency"], []).append(r["ttft_ms"])
    for k in sorted(h):
        v = sorted(h[k])
        if len(v) < 5:
            continue
        ratio = f"{statistics.median(v)/ref:>8.2f}x" if ref else "     n/a"
        print(f"  {k:>6} {len(v):>5} {statistics.median(v):>9.0f} "
              f"{v[min(len(v)-1, int(.99*(len(v)-1)))]:>9.0f} {ratio}")

    # does TTFT track KV size, as a transfer-bound cost should?
    print(f"\n  {'KV bucket':>14} {'n':>5} {'p50 TTFT':>10}")
    edges = [0.2, 0.4, 0.7, 1.0, 2.0]
    prev = 0.0
    for e in edges:
        v = sorted(r["ttft_ms"] for r in warm if prev <= r["kv_gb"] < e)
        if len(v) >= 5:
            print(f"  {prev:>5.2f}-{e:<5.2f} GB {len(v):>5} "
                  f"{statistics.median(v):>9.0f}")
        prev = e

    gb = (
        (m1["bytes_c2g"] - m0["bytes_c2g"]) / GB
        if "bytes_c2g" in m0 and "bytes_c2g" in m1
        else None
    )
    sec = (
        m1["time_c2g"] - m0["time_c2g"]
        if "time_c2g" in m0 and "time_c2g" in m1
        else None
    )
    if gb is not None and sec is not None:
        print(f"\n  PCIe reload : {gb:.0f} GB in {sec:.1f}s "
              f"({gb / max(sec, 1e-9):.1f} GB/s) = {100 * sec / E:.0f}% of wall")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"args": {k: v for k, v in vars(a).items()
                                if isinstance(v, (str, int, float, bool))},
                       "elapsed_s": round(E, 2), "baseline_ttft_ms": ref,
                       "pcie": None if gb is None or sec is None else {
                           "reload_gb": round(gb, 2),
                           "reload_s": round(sec, 2)},
                       "rows": rows, "errors": errors,
                       "undrained_sessions": undrained,
                       "goodput": {
                           "slo_ms": a.goodput_slo_ms,
                           "successes": good_count,
                           "total": n,
                           "rate": round(good_count / n, 6),
                       },
                       "goodput_policy": (
                           goodput.snapshot_stats() if goodput else None
                       ),
                       "selective_shedding_policy": (
                           shedding.snapshot_stats() if shedding else None
                       ),
                       "pressure_eviction_policy": (
                           pressure.snapshot_stats() if pressure else None
                       ),
                       "deadline_floor_policy": (
                           deadline_floor.snapshot_stats()
                           if deadline_floor else None
                       ),
                       "service_curve_edf_policy": (
                           service_curve_edf.snapshot_stats()
                           if service_curve_edf else None
                       ),
                       "shared_bandwidth_policy": (
                           shared_bandwidth.snapshot_stats()
                           if shared_bandwidth else None
                       )}, f, indent=2)
        print(f"  wrote {a.json}")


if __name__ == "__main__":
    main()
