"""Watch one run; print a report and exit at the first anomaly, at the end, or after --every seconds, whichever is first.

python watch.py <app_id> <run_id> --started-epoch E [--every 900]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

RATE_PER_H = 42.4
LIMIT = 105.0  # this attempt; the earlier ones took about $85 of the $200
STALL_SECS = 900
PATTERN = re.compile(r"\w(Error|Exception): |LoRA weight|weight.*mismatch|not equal|CUDA out of memory|illegal address|evicted|tearing down|FATAL|Killed|left the pool", re.I)
BENIGN = re.compile(r"metrics.py:|perf [0-9]+:|freeze_gc|Errno 111|ClientDisconnect|retrying\.\.\.|wait_init_expected|Failed to establish a new connection.*:3200\d|FutureWarning|UserWarning")  # engine startup
TOKENS = re.compile(r"(?i)(token[=:/ ]+)\S+")


def sh(*argv, timeout=60) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired as e:
        out = e.stdout or b""
        return out.decode(errors="replace") if isinstance(out, bytes) else out


def containers(app_id: str) -> int:
    out = sh(sys.executable, "-m", "modal", "container", "list", "--json")
    try:
        return sum(c["app_id"] == app_id for c in json.loads(out))
    except ValueError:
        return -1


def logs(app_id: str) -> list[str]:
    """What the app printed in the last while; `app logs` follows, so it is cut off by the timeout."""
    return sh("timeout", "25", sys.executable, "-m", "modal", "app", "logs", app_id, timeout=30).splitlines()


def metrics(run_id: str) -> dict:
    env = Path(os.environ["MILES_DOTENV_DIR"], ".env").read_text().splitlines()
    os.environ["WANDB_API_KEY"] = next(
        line.split("=", 1)[1].strip().strip("\"'") for line in env if line.startswith("WANDB_API_KEY=")
    )
    import wandb

    site = json.loads((Path(__file__).parent / "site.json").read_text())
    api = wandb.Api(timeout=30, overrides={"base_url": site["wandb_base_url"]})
    runs = list(api.runs(site["wandb_project"], filters={"group": run_id})) or list(
        api.runs(site["wandb_project"], filters={"display_name": run_id})
    )
    if not runs:
        return {}
    run = runs[0]
    wanted = ["rollout/raw_reward", "rollout/truncated", "rollout/response_lengths", "train/train_rollout_logprob_abs_diff",
              "train/grad_norm", "train/step", "_timestamp"]
    rows = [r for r in run.history(keys=["rollout/raw_reward", "_timestamp"], samples=400, pandas=False)]
    rewards = [r["rollout/raw_reward"] for r in rows if r.get("rollout/raw_reward") is not None]
    times = [r["_timestamp"] for r in rows]
    gaps = [b - a for a, b in zip(times, times[1:])]
    return {"url": run.url, "state": run.state, "step": len(rewards), "recent_rewards": [round(x, 3) for x in rewards[-10:]],
            "first_10_mean": round(sum(rewards[:10]) / max(1, len(rewards[:10])), 3),
            "last_10_mean": round(sum(rewards[-10:]) / max(1, len(rewards[-10:])), 3),
            "secs_per_step_last_10": round(sum(gaps[-10:]) / max(1, len(gaps[-10:])), 1),
            "trunc": [round(r["rollout/truncated"], 3) for r in run.scan_history(keys=["rollout/truncated"])][-12:],
            "gap": [round(r["train/train_rollout_logprob_abs_diff"], 4) for r in run.scan_history(keys=["train/train_rollout_logprob_abs_diff"])][-12:],
            "summary": {k: run.summary.get(k) for k in wanted}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("app_id")
    ap.add_argument("run_id")
    ap.add_argument("--started-epoch", type=float, required=True)
    ap.add_argument("--every", type=int, default=900)
    a = ap.parse_args()
    start, last_step, last_change, seen = time.time(), None, time.time(), set()
    reason = "tick"
    while time.time() - start < a.every:
        time.sleep(60)
        hits = [l for l in logs(a.app_id) if PATTERN.search(l) and not BENIGN.search(l)]
        new = [l for l in hits if l not in seen]
        seen.update(hits)
        n = containers(a.app_id)
        cost = (time.time() - a.started_epoch) / 3600 * RATE_PER_H
        try:
            m = metrics(a.run_id)
        except Exception as e:
            m = {"error": str(e)[:200]}
        if m.get("step") != last_step:
            last_step, last_change = m.get("step"), time.time()
        if new:
            reason = "log anomaly"
            break
        gap, trunc = (m.get("gap") or [])[-5:], (m.get("trunc") or [])[-5:]
        if len(gap) == 5 and sum(gap) / 5 > 0.025:
            reason = "log-prob gap over 0.025 for 5 steps"
            break
        if len(trunc) == 5 and sum(trunc) / 5 > 0.2:
            reason = "over 20% of replies capped for 5 steps"
            break
        if n == 0:
            reason = "no containers left"
            break
        if cost > LIMIT:
            reason = "budget"
            break
        if last_step is not None and time.time() - last_change > STALL_SECS:
            reason = "no training step for 15 min"
            break
    print(f"reason: {reason}; containers {n}; cost so far about ${cost:.0f}; wandb step {last_step}")
    for line in new[-25:]:
        print(TOKENS.sub(r"\1***", line)[:300])
    print(json.dumps(m, indent=0, default=str)[:6000])


main()
