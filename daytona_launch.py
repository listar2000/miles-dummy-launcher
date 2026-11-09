"""The makeshift launcher with its rollout node on Daytona instead of Modal; the trainer stays on Modal.

    .venv-daytona/bin/python daytona_launch.py probe-cpu          preflight B: outbound access and the preview proxy
    .venv-daytona/bin/python daytona_launch.py probe-gpu          preflight C: image, GPUs, weights, engines, pool load
    .venv-daytona/bin/python daytona_launch.py run --run-id X     the run: Daytona rollout node first, then the trainer

Everything created in Daytona is named and labelled miles-rl-*; nothing else in the org is listed or touched by name.
The GCS login reuses a Modal identity token of this app, written to a file in the sandbox, never an env var or a log.
"""

import argparse
import concurrent.futures
import json
import os
import secrets
import shlex
import sys
import time
import urllib.request
from pathlib import Path

import modal
if modal.is_local():  # the Modal helpers below run in the image, which has no Daytona SDK
    from daytona import (
        CreateSandboxFromImageParams,
        Daytona,
        DaytonaConfig,
        GpuType,
        Resources,
        SessionExecuteRequest,
    )

import launcher as L

HERE = Path(__file__).parent
PREFIX = "miles-rl-"
LABELS = {"owner": "miles-rl", "purpose": "e2e"}
ROLLOUT_PORT = 32999
NODE_DIR = "/root/node"  # spec, token and login config inside the sandbox
SIGNED_URL_SECS = 86400  # Daytona's maximum
HELPER_IMAGE = L.image.add_local_python_source("launcher")  # this module imports it in the container too


def _env_file_value(name: str) -> str:
    for line in Path(os.environ["MILES_DOTENV_DIR"], ".env").read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise KeyError(name)


def client(target: str | None = None):
    return Daytona(DaytonaConfig(api_key=_env_file_value("DAYTONA_API_KEY"), target=target))


# --- Modal helpers, in the launcher's app so the identity token names it ---------------------------------------


@L.app.function(image=HELPER_IMAGE, cpu=1, memory=1024, timeout=1800)
def _echo_tunnel(seconds: int, key: str) -> None:
    """A Modal tunnel to a server that answers /sleep/<n> after n seconds: what a Daytona node's beats would reach."""
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            n = int(self.path.rsplit("/", 1)[-1]) if self.path.startswith("/sleep/") else 0
            time.sleep(n)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("0.0.0.0", 30999), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with modal.forward(30999) as t:
        L.state[key] = t.url
        time.sleep(seconds)


@L.app.function(image=HELPER_IMAGE, cpu=2, memory=2048, timeout=1800)
def _hammer(url: str, n: int, secs: int) -> dict:
    """n concurrent GETs of <url>/sleep/<secs> from Modal: the trainer calling a rollout node through its preview URL."""

    def one(i):
        t = time.time()
        try:
            with urllib.request.urlopen(f"{url.rstrip('/')}/sleep/{secs}", timeout=secs + 120) as r:
                return r.status, round(time.time() - t, 1)
        except Exception as e:
            return f"{type(e).__name__}: {str(e)[:80]}", round(time.time() - t, 1)

    with concurrent.futures.ThreadPoolExecutor(n) as pool:
        results = list(pool.map(one, range(n)))
    statuses = {}
    for s, _ in results:
        statuses[str(s)] = statuses.get(str(s), 0) + 1
    return {"statuses": statuses, "max_secs": max(d for _, d in results), "min_secs": min(d for _, d in results)}


@L.app.function(image=HELPER_IMAGE, cpu=1, memory=1024, timeout=300)
def _identity_token() -> str:
    return os.environ["MODAL_IDENTITY_TOKEN"]


# --- Daytona -------------------------------------------------------------------------------------------------


def create(d, name: str, image: str, resources, ttl_minutes: int, target: str | None):
    params = CreateSandboxFromImageParams(
        name=name,
        image=image,
        resources=resources,
        labels=LABELS,
        auto_stop_interval=0,  # GPU sandboxes are deleted when they stop; the TTL bounds the bill instead
        ttl_minutes=ttl_minutes,
        target=target,
    )
    return d.create(params, timeout=0, on_snapshot_create_logs=lambda line: print(f"[image] {line}", flush=True))


def teardown(sandbox) -> None:
    """Stop: this key may not delete, and Daytona deletes a GPU sandbox when it stops."""
    try:
        sandbox.stop()
    except Exception as e:
        print(f"stop failed: {type(e).__name__}: {str(e)[:200]}; the sandbox TTL still ends it", flush=True)


def sh(sandbox, cmd: str, timeout: int = 120) -> tuple[int, str]:
    r = sandbox.process.exec(cmd, timeout=timeout)
    return r.exit_code, (r.result or "")


def upload(sandbox, path: str, data: bytes) -> None:
    sandbox.fs.upload_file(data, path)


def signed_url(sandbox, port: int, secs: int = SIGNED_URL_SECS) -> str:
    return sandbox.create_signed_preview_url(port, expires_in_seconds=secs).url


def gcp_login(sandbox, token: str) -> dict:
    """Write the token and the login config as files; return the env that points at them (no secret in it)."""
    token_path, config_path = f"{NODE_DIR}/modal-oidc-token", f"{NODE_DIR}/gcp-wif.json"
    sh(sandbox, f"mkdir -p {NODE_DIR} && chmod 700 {NODE_DIR}")
    upload(sandbox, token_path, token.encode())
    pool = (
        f"projects/{L.GCP_PROJECT_NUMBER}/locations/global/workloadIdentityPools/{L.GCP_POOL}/providers/{L.GCP_PROVIDER}"
    )
    config = {
        "type": "external_account",
        "audience": f"//iam.googleapis.com/{pool}",
        "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
        "token_url": "https://sts.googleapis.com/v1/token",
        "credential_source": {"file": token_path},
        "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/"
        f"serviceAccounts/{L.GCP_ACCOUNT}:generateAccessToken",
    }
    upload(sandbox, config_path, json.dumps(config).encode())
    sh(sandbox, f"chmod 600 {token_path} {config_path}")
    return {"GOOGLE_APPLICATION_CREDENTIALS": config_path, "GOOGLE_CLOUD_PROJECT": L.GCP_PROJECT}


# --- preflight B ---------------------------------------------------------------------------------------------

_EGRESS = r"""
import json, sys, urllib.request
out = {}
for name, url in json.loads(sys.argv[1]).items():
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            out[name] = r.status
    except urllib.error.HTTPError as e:
        out[name] = e.code  # an HTTP answer still means the host is reachable
    except Exception as e:
        out[name] = type(e).__name__ + ": " + str(e)[:80]
print(json.dumps(out))
"""

_SLOW_SERVER = r"""
import http.server, time
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        n = int(self.path.rsplit("/", 1)[-1]) if self.path.startswith("/sleep/") else 0
        time.sleep(n); self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("0.0.0.0", 32999), H).serve_forever()
"""


def probe_cpu(args) -> None:
    d = client(args.target)
    with modal.enable_output(), L.app.run():
        key = f"probe/echo-url/{secrets.token_hex(3)}"
        echo = _echo_tunnel.spawn(900, key)
        name = f"{PREFIX}e2e-probe-cpu-{secrets.token_hex(3)}"
        sandbox = create(d, name, "python:3.11.9-slim", None, 30, args.target)
        print(f"sandbox {name} {sandbox.id} target {sandbox.target}", flush=True)
        try:
            for _ in range(150):
                if echo_url := L.state.get(key):
                    break
                time.sleep(2)
            else:
                raise RuntimeError("the Modal echo tunnel never came up")
            upload(sandbox, "/tmp/egress.py", _EGRESS.encode())
            hosts = {
                "gcs": "https://storage.googleapis.com/",
                "sts": "https://sts.googleapis.com/",
                "huggingface": "https://huggingface.co/api/models/Qwen/Qwen3-8B",
                "wandb": L.SITE["wandb_base_url"] + "/",
                "docker_hub": "https://registry-1.docker.io/v2/",
                "modal_tunnel": echo_url + "/sleep/0",
            }
            print("egress", sh(sandbox, f"python /tmp/egress.py {shlex.quote(json.dumps(hosts))}", 120), flush=True)
            # a beat that waits: the registry answers beats at once, but a slow one must not be cut
            hosts = {"modal_tunnel_60s": echo_url + "/sleep/60"}
            print("slow egress", sh(sandbox, f"python /tmp/egress.py {shlex.quote(json.dumps(hosts))}", 180), flush=True)
            upload(sandbox, "/tmp/slow.py", _SLOW_SERVER.encode())
            sandbox.process.create_session("slow")
            sandbox.process.execute_session_command(
                "slow", SessionExecuteRequest(command="python /tmp/slow.py", run_async=True)
            )
            time.sleep(3)
            url = signed_url(sandbox, ROLLOUT_PORT, 3600)
            for n, secs in [(int(x.split("x")[0]), int(x.split("x")[1])) for x in args.preview_tests.split(",")]:
                print(f"preview {n} req {secs}s", _hammer.remote(url, n, secs), flush=True)
        finally:
            teardown(sandbox)
            echo.cancel()
            print(f"stopped {name}", flush=True)


# --- preflight C ---------------------------------------------------------------------------------------------


def gpu_resources():
    return Resources(gpu=4, gpu_type=GpuType.H200, cpu=16, memory=192, disk=512)


def probe_gpu(args) -> None:
    d = client(args.target)
    image = L.IMAGE
    with modal.enable_output(), L.app.run():
        token = _identity_token.remote()
        name = f"{PREFIX}e2e-probe-gpu-{secrets.token_hex(3)}"
        t0 = time.time()
        sandbox = create(d, name, image, gpu_resources(), 60, args.target)
        print(f"sandbox {name} {sandbox.id} target {sandbox.target} ready after {time.time() - t0:.0f}s", flush=True)
        try:
            for cmd in (
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                "df -h /dev/shm / | tail -2; nproc; free -g | head -2",
                "python -c 'import miles_ext, sglang; print(sglang.__version__)'",
            ):
                print(cmd.split()[0], sh(sandbox, cmd, 120), flush=True)
            env = gcp_login(sandbox, token)
            env_s = " ".join(f"{k}={v}" for k, v in env.items())
            gcs = (
                f"{env_s} python -c \"from google.cloud import storage; c=storage.Client(); "
                f"b=c.bucket('{L.BUCKET}'); bl=b.blob('{L.GCS_FOLDER}/_probe/daytona.txt'); "
                "bl.upload_from_string('ok'); print(bl.download_as_text()); bl.delete()\""
            )
            print("gcs", sh(sandbox, gcs, 120), flush=True)
            t = time.time()
            prep = (
                "python -m miles_ext.node prepare-weights --hf-repo Qwen/Qwen3-8B "
                "--revision b968826d9c46dd6066d109eabc6255188de91218 --path /models/Qwen3-8B 2>&1 | tail -2"
            )
            print("weights", sh(sandbox, prep, 1800), f"{time.time() - t:.0f}s", flush=True)
            # two engines the way the rollout role starts them, then real requests and the pool numbers (#205)
            engines = (
                "cd /root && for i in 0 1; do CUDA_VISIBLE_DEVICES=$i nohup python -m sglang.launch_server "
                "--model-path /models/Qwen3-8B --port $((32000+i)) --enable-lora --max-lora-rank 16 "
                "> /tmp/engine$i.log 2>&1 & done; sleep 1; echo started"
            )
            print("engines", sh(sandbox, engines, 60), flush=True)
            t = time.time()
            for _ in range(120):
                code, out = sh(sandbox, "curl -s -o /dev/null -w '%{http_code}' localhost:32000/health; echo", 30)
                if out.strip().startswith("200"):
                    break
                time.sleep(10)
            print("engine health", out.strip(), f"{time.time() - t:.0f}s", flush=True)
            load = (
                "for k in $(seq 24); do curl -s localhost:32000/generate -H 'Content-Type: application/json' "
                "-d '{\"text\":\"Count slowly to two hundred.\",\"sampling_params\":{\"max_new_tokens\":800}}' "
                ">/dev/null & done; sleep 4; python -c \"import sys; sys.path.insert(0, '/root/miles'); "
                "from miles_ext.node import rollout; print(rollout._load([32000, 32001]))\""
            )
            print("pool load", sh(sandbox, load, 120), flush=True)
        finally:
            teardown(sandbox)
            print(f"stopped {name} after {time.time() - t0:.0f}s", flush=True)


# --- the run -------------------------------------------------------------------------------------------------


def _node_env(run_id: str, pool_token: str, public_url: str) -> dict:
    """The rollout node's env, written to a 0600 file and sourced by its command, so no secret is sandbox metadata."""
    home = Path.home() / "miles-ui-data"
    return {
        "PYTHONUNBUFFERED": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NCCL_DEBUG": "WARN",
        "MILES_GPU": "H200",
        "MILES_STORAGE": "posix",  # /results stays local: with a weight sync mount the rollout role reads nothing there
        "MILES_STATE": "http",
        "MILES_STATE_URL": (home / "url.txt").read_text().strip() + "/ingest",
        "MILES_STATE_TOKEN": (home / "ingest.token").read_text().strip(),
        "MILES_WEIGHT_SYNC_MOUNT": "/weight-sync",
        "MILES_GCS_MOUNTS": json.dumps({"/weight-sync": f"gs://{L.BUCKET}/{L.GCS_FOLDER}/{run_id}"}),
        "MILES_POOL_TOKEN": pool_token,
        "MILES_TRAINER_URL_FILE": f"{NODE_DIR}/trainer-url",
        "MILES_ROLLOUT_PUBLIC_URL": public_url,
        "WANDB_API_KEY": _env_file_value("WANDB_API_KEY"),
    }


def _follow(sandbox, session: str, cmd_id: str, path: Path) -> None:
    """Copy the node's output to a local file: a GPU sandbox's disk goes when it is deleted."""
    import asyncio
    import re

    redact = re.compile(r"(https://\d+-)[A-Za-z0-9]+(\.)|((?i:token)[=: ]+)\S+")

    def write(line: str) -> None:
        with path.open("a") as f:
            f.write(redact.sub(lambda m: f"{m.group(1)}***{m.group(2)}" if m.group(1) else f"{m.group(3)}***", line))
            if not line.endswith("\n"):
                f.write("\n")

    try:
        result = sandbox.process.get_session_command_logs_async(session, cmd_id, write, write)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    except Exception as e:
        write(f"[launcher] log stream ended: {type(e).__name__}: {e}")


def run(args) -> None:
    import threading

    run_id = args.run_id
    log = HERE / "logs" / f"{run_id}-daytona.log"
    d = client(args.target)
    with modal.enable_output(), L.app.run(detach=True):
        spec, manifest = L.make_spec.remote(0, run_id)
        roles = manifest["roles"]
        print(json.dumps(roles, indent=1), flush=True)
        role = roles["rollout"]
        token = _identity_token.remote()
        pool_token = secrets.token_urlsafe(24)
        name = f"{PREFIX}{run_id}"
        resources = Resources(
            gpu=role["gpus_per_node"], gpu_type=GpuType.H200, cpu=role["cpu"], memory=role["memory_gib"], disk=512
        )
        t0 = time.time()
        sandbox = create(d, name, L.IMAGE, resources, args.ttl_minutes, args.target)
        print(f"sandbox {name} {sandbox.id} target {sandbox.target} after {time.time() - t0:.0f}s", flush=True)
        trainer_call, outcome = None, {}
        try:
            env = {**_node_env(run_id, pool_token, signed_url(sandbox, ROLLOUT_PORT)), **gcp_login(sandbox, token)}
            upload(sandbox, f"{NODE_DIR}/env", "".join(f"{k}={shlex.quote(v)}\n" for k, v in env.items()).encode())
            upload(sandbox, f"{NODE_DIR}/spec.json", json.dumps(spec).encode())
            sh(sandbox, f"chmod 600 {NODE_DIR}/env && mkdir -p /results /weight-sync /models")
            command = (
                f"set -a; . {NODE_DIR}/env; set +a; cd /root/miles && "
                "python -m miles_ext.node prepare-weights --hf-repo Qwen/Qwen3-8B "
                "--revision b968826d9c46dd6066d109eabc6255188de91218 --path /models/Qwen3-8B && "
                f"exec python -m miles_ext.node rollout --spec {NODE_DIR}/spec.json"
            )
            sandbox.process.create_session("node")
            cmd_id = sandbox.process.execute_session_command(
                "node", SessionExecuteRequest(command=command, run_async=True)
            ).cmd_id
            threading.Thread(target=_follow, args=(sandbox, "node", cmd_id, log), daemon=True).start()
            # the rollout node runs: start the trainer now, so both load together (fork/guide/launcher.md)
            trainer_call = L.trainer.with_options(**L._options(roles["trainer"])).spawn(spec, pool_token)
            print(f"trainer call {trainer_call.object_id}", flush=True)
            L.state[f"{run_id}/calls"] = {"trainer": trainer_call.object_id, "daytona": sandbox.id}
            seen_url = None
            while True:
                url = L.state.get(f"{run_id}/trainer-url")
                if url and url != seen_url:
                    upload(sandbox, f"{NODE_DIR}/trainer-url", url.encode())
                    seen_url = url
                done, value = L._finished(trainer_call)
                if done:
                    outcome = {"ended": "trainer", "value": value}
                    break
                exit_code = sandbox.process.get_session_command("node", cmd_id).exit_code
                if exit_code is not None:
                    outcome = {"ended": "rollout", "value": exit_code}
                    break
                time.sleep(15)
        finally:
            print(f"tearing down: {outcome}", flush=True)
            if trainer_call is not None:
                trainer_call.cancel(terminate_containers=True)
            teardown(sandbox)
            L.state[f"{run_id}/outcome"] = outcome
            print(f"stopped {name} after {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe-cpu", "probe-gpu", "run"])
    ap.add_argument("--target", default=None)
    ap.add_argument("--run-id")
    ap.add_argument("--ttl-minutes", type=int, default=270)
    ap.add_argument("--preview-tests", default="1x5,64x300")
    args = ap.parse_args()
    {"probe-cpu": probe_cpu, "probe-gpu": probe_gpu, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
