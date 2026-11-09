"""A makeshift outer launcher for one Miles run on Modal: it knows the image, the manifest and fork/guide/launcher.md.

    python -m modal run launcher.py::ident                 the workspace id and token lifetime, for the GCP setup
    python -m modal run launcher.py::dry                   CPU checks in the pinned image
    python -m modal run --detach launcher.py::launch --run-id X [--num-rollout N]

A driver function places the nodes and tears them down. Nodes run as subprocesses, the way any other provider would.
"""

import base64
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import modal

IMAGE = "docker.io/starlimobius/miles-mobius@sha256:d0f97890e3e3bf4505d88bf6ab37231cb6806513a089f4bf11151322fde95936"
APP = "miles-rl-e2e"
RESULTS_VOLUME, WEIGHTS_VOLUME, CACHE_VOLUME = "miles-rl-results", "miles-rl-weights", "miles-compile-cache"
STATE_DICT = "miles-rl-e2e-state"
# where this launcher's cloud resources are: site.json locally (copy site.example.json), the image env in containers
SITE_ENV = "E2E_SITE"
SITE = json.loads(os.environ.get(SITE_ENV) or (Path(__file__).parent / "site.json").read_text())
BUCKET, GCS_FOLDER = SITE["bucket"], SITE["gcs_folder"]
GCP_PROJECT, GCP_PROJECT_NUMBER = SITE["gcp_project"], SITE["gcp_project_number"]
GCP_POOL, GCP_PROVIDER, GCP_ACCOUNT = SITE["gcp_pool"], SITE["gcp_provider"], SITE["gcp_account"]
REGISTRY_PORT, ROLLOUT_PORT = 30999, 32999  # the manifest lists them; forward() needs the ones it names
TRAINER_URL_FILE = "/tmp/miles-trainer-url"
SPEC_FILE = "/tmp/spec.json"
FIRST_NODE_TIMEOUT_SECS = 1800
HOURS = 3600
LOCAL = Path(__file__).parent


def _wandb_secret() -> modal.Secret:
    """Only the key the manifest asks for, not the rest of the box's .env."""
    if not modal.is_local():
        return modal.Secret.from_dict({})
    env = Path(os.environ["MILES_DOTENV_DIR"], ".env").read_text().splitlines()
    key = next(line.split("=", 1)[1].strip().strip("\"'") for line in env if line.startswith("WANDB_API_KEY="))
    home = Path.home() / "miles-ui-data"
    return modal.Secret.from_dict(
        {
            "WANDB_API_KEY": key,
            "MILES_STATE_URL": (home / "url.txt").read_text().strip() + "/ingest",
            "MILES_STATE_TOKEN": (home / "ingest.token").read_text().strip(),
        }
    )


app = modal.App(APP)
image = (
    modal.Image.from_registry(IMAGE)
    .entrypoint([])
    .env({"PYTHONUNBUFFERED": "1", "NCCL_NVLS_ENABLE": "0", "NCCL_DEBUG": "WARN", SITE_ENV: json.dumps(SITE)})
    .add_local_file(LOCAL / "e2e_tasks/__init__.py", "/root/miles/e2e_tasks/__init__.py")
    .add_local_file(LOCAL / "e2e_tasks/math_grade.py", "/root/miles/e2e_tasks/math_grade.py")
    .add_local_file(LOCAL / "recipe.yaml", "/e2e/recipe.yaml")
    .add_local_file(LOCAL / "prompts.jsonl", "/e2e/prompts.jsonl")
    .add_local_file(LOCAL / "prompts.jsonl", "/inputs/prompts.jsonl")
)
volumes = {
    "/models": modal.Volume.from_name(WEIGHTS_VOLUME).read_only(),
    "/results": modal.Volume.from_name(RESULTS_VOLUME),
    "/compile-cache": modal.Volume.from_name(CACHE_VOLUME),
}
state = modal.Dict.from_name(STATE_DICT, create_if_missing=True)


def _gcp_login() -> str:
    """Keyless: exchange this container's Modal identity token for the service account, through the ADC file."""
    token, config = Path("/tmp/modal-oidc-token"), Path("/tmp/gcp-wif.json")
    token.write_text(os.environ["MODAL_IDENTITY_TOKEN"])
    token.chmod(0o600)
    pool = f"projects/{GCP_PROJECT_NUMBER}/locations/global/workloadIdentityPools/{GCP_POOL}/providers/{GCP_PROVIDER}"
    config.write_text(
        json.dumps(
            {
                "type": "external_account",
                "audience": f"//iam.googleapis.com/{pool}",
                "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                "token_url": "https://sts.googleapis.com/v1/token",
                "credential_source": {"file": str(token)},
                "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/"
                f"serviceAccounts/{GCP_ACCOUNT}:generateAccessToken",
            }
        )
    )
    return str(config)


def _node_env(run_id: str) -> dict:
    os.makedirs("/weight-sync", exist_ok=True)
    return {
        **os.environ,
        "MILES_STORAGE": "modal-volume",
        "MILES_MODAL_VOLUMES": json.dumps(
            {"/results": RESULTS_VOLUME, "/models": WEIGHTS_VOLUME, "/compile-cache": CACHE_VOLUME}
        ),
        "MILES_STATE": "http" if os.environ.get("MILES_STATE_URL") else "modal-dict",
        "MILES_MODAL_DICT": json.dumps({"name": STATE_DICT, "environment": os.environ.get("MODAL_ENVIRONMENT")}),
        "MILES_WEIGHT_SYNC_MOUNT": "/weight-sync",
        "MILES_GCS_MOUNTS": json.dumps({"/weight-sync": f"gs://{BUCKET}/{GCS_FOLDER}/{run_id}"}),
        "GOOGLE_APPLICATION_CREDENTIALS": _gcp_login(),
        "GOOGLE_CLOUD_PROJECT": GCP_PROJECT,  # else google.auth asks Resource Manager for it
    }


def _node(role: str, spec: dict, env: dict) -> int:
    Path(SPEC_FILE).write_text(json.dumps(spec))
    return subprocess.run([sys.executable, "-m", "miles_ext.node", role, "--spec", SPEC_FILE], env=env).returncode


@app.function(image=image, volumes=volumes, cpu=2, memory=8192, timeout=900)
def make_spec(num_rollout: int, run_id: str) -> tuple[dict, dict]:
    argv = [sys.executable, "-m", "miles_ext.node", "spec", "--recipe", "/e2e/recipe.yaml", "--run-id", run_id]
    if num_rollout:
        argv += ["--num-rollout", str(num_rollout)]
    spec = json.loads(subprocess.run(argv, capture_output=True, text=True, check=True).stdout)
    Path(SPEC_FILE).write_text(json.dumps(spec))
    manifest = subprocess.run(
        [sys.executable, "-m", "miles_ext.node", "manifest", "--spec", SPEC_FILE],
        capture_output=True, text=True, check=True,
    ).stdout
    return spec, json.loads(manifest)


@app.function(image=image, volumes={"/models": modal.Volume.from_name(WEIGHTS_VOLUME)}, cpu=4, memory=16384, timeout=3600)
def _prepare_weights(hf_repo: str, revision: str, path: str) -> int:
    rc = subprocess.run(
        [sys.executable, "-m", "miles_ext.node", "prepare-weights", "--hf-repo", hf_repo, "--revision", revision,
         "--path", path],
        env={**os.environ, "MILES_STORAGE": "modal-volume", "MILES_MODAL_VOLUMES": json.dumps({"/models": WEIGHTS_VOLUME})},
    ).returncode
    modal.Volume.from_name(WEIGHTS_VOLUME).commit()
    return rc


@app.local_entrypoint()
def prepare_weights(hf_repo: str, revision: str, path: str):
    print("rc", _prepare_weights.remote(hf_repo, revision, path))


@app.function(image=image, cpu=1, memory=2048, timeout=300)
def _ident() -> dict:
    payload = os.environ["MODAL_IDENTITY_TOKEN"].split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    return {
        "workspace_id": claims["workspace_id"],
        "app_name": claims["app_name"],
        "aud": claims["aud"],
        "iss": claims["iss"],
        "lifetime_h": (claims["exp"] - claims["iat"]) / 3600,
        "sub_len": len(claims["sub"]),
    }


@app.local_entrypoint()
def ident():
    print(_ident.remote())


_DRY_STORAGE = r"""
import hashlib, os, pathlib, time
from miles_ext.node.adapters.gcs import mirror
from miles_ext.node.storage import for_mount

# the results volume, committed from a subprocess
r = pathlib.Path("/results/_e2e-probe"); r.mkdir(exist_ok=True); (r / "marker").write_text(str(time.time()))
results = for_mount("/results"); results.commit(); results.refresh(); print("results volume commit and reload: ok")

# weight-sized round trip through the bucket, keyless
data = os.urandom(100 * 2**20); digest = hashlib.sha256(data).hexdigest()
a = for_mount("/weight-sync"); p = pathlib.Path("/weight-sync/probe.bin"); p.write_bytes(data)
t = time.time(); a.commit(); print(f"upload 100 MiB: {time.time() - t:.1f}s")
b = mirror(pathlib.Path("/tmp/ws2"), os.environ["PROBE_URI"]); t = time.time(); b.refresh()
print(f"download 100 MiB: {time.time() - t:.1f}s")
assert hashlib.sha256(pathlib.Path("/tmp/ws2/probe.bin").read_bytes()).hexdigest() == digest; print("round trip: ok")
p.unlink(); a.commit(); b.refresh(); assert not pathlib.Path("/tmp/ws2/probe.bin").exists(); print("delete: ok")
"""

_DRY_GRADER = r"""
import json
from miles_ext.tasks.single_turn import check_row, _verifier
row = json.loads(open("/inputs/prompts.jsonl").readline()); check_row(row)
g = _verifier(row["metadata"]["task"])
for reply, label in [("x\nAnswer: \\boxed{34}", "34"), ("x\nAnswer: \\boxed{35}", "34"), ("no final line", "34")]:
    print(repr(reply), label, g(reply, label))
"""


@app.function(image=image, volumes=volumes, cpu=4, memory=16384, timeout=1800)
def _dry(spec: dict) -> dict:
    env = _node_env("_probe")
    results = {}
    results["grader"] = subprocess.run([sys.executable, "-c", _DRY_GRADER], env=env, capture_output=True, text=True)
    results["storage"] = subprocess.run(
        [sys.executable, "-c", _DRY_STORAGE],
        env={**env, "PROBE_URI": f"gs://{BUCKET}/{GCS_FOLDER}/_probe"}, capture_output=True, text=True,
    )
    Path(SPEC_FILE).write_text(json.dumps(spec))
    results["check"] = subprocess.run(
        [sys.executable, "-m", "miles_ext.node", "check", "--spec", SPEC_FILE, "--no-gpu"],
        env=env, capture_output=True, text=True,
    )
    return {k: {"rc": v.returncode, "out": v.stdout[-3000:], "err": v.stderr[-3000:]} for k, v in results.items()}


@app.local_entrypoint()
def dry():
    spec, manifest = make_spec.remote(0, "e2e-dry")
    print(json.dumps(manifest, indent=1))
    for name, r in _dry.remote(spec).items():
        print(f"--- {name}: rc {r['rc']}\n{r['out']}\n{r['err']}")


@app.function(image=image, volumes=volumes, secrets=[_wandb_secret()], cpu=1, memory=1024, timeout=5 * HOURS,
              single_use_containers=True)
def rollout(spec: dict, pool_token: str, index: int) -> int:
    import threading

    run_id = spec["run_id"]
    state[f"{run_id}/rollout-{index}"] = {"phase": "container", "at": time.time()}

    def carry_url() -> None:
        seen = None
        while True:
            url = state.get(f"{run_id}/trainer-url")
            if url and url != seen:
                Path(TRAINER_URL_FILE + ".tmp").write_text(url)
                os.replace(TRAINER_URL_FILE + ".tmp", TRAINER_URL_FILE)
                seen = url
            time.sleep(5)

    threading.Thread(target=carry_url, daemon=True).start()
    with modal.forward(ROLLOUT_PORT) as tunnel:
        return _node(
            "rollout",
            spec,
            {
                **_node_env(run_id),
                "MILES_POOL_TOKEN": pool_token,
                "MILES_TRAINER_URL_FILE": TRAINER_URL_FILE,
                "MILES_ROLLOUT_PUBLIC_URL": tunnel.url,
            },
        )


@app.function(image=image, volumes=volumes, secrets=[_wandb_secret()], cpu=1, memory=1024, timeout=5 * HOURS,
              single_use_containers=True)
def trainer(spec: dict, pool_token: str) -> dict:
    run_id = spec["run_id"]
    with modal.forward(REGISTRY_PORT) as tunnel:
        state[f"{run_id}/trainer-url"] = tunnel.url
        rc = _node(
            "trainer",
            spec,
            {
                **_node_env(run_id),
                "MILES_NODE_RANK": "0",
                "MILES_NNODES": "1",
                "MILES_HEAD_ADDR": "127.0.0.1",
                "MILES_NODE_IP": "127.0.0.1",
                "MILES_POOL_TOKEN": pool_token,
                "MILES_FIRST_ROLLOUT_NODE_TIMEOUT_SECS": str(FIRST_NODE_TIMEOUT_SECS),
                "MILES_CALL_ID": modal.current_function_call_id(),
            },
        )
    volumes["/results"].reload()
    exit_file = Path(f"/results/{run_id}/exit.json")
    return {"rc": rc, "exit": json.loads(exit_file.read_text()) if exit_file.exists() else None}


def _options(role: dict) -> dict:
    return {
        "gpu": f"{role.get('gpu') or role['gpu_advice']}:{role['gpus_per_node']}",
        "cpu": role["cpu"],
        "memory": role["memory_gib"] * 1024,
        "timeout": int(role["timeout_hours"] * HOURS),
    }


def _finished(call: modal.FunctionCall) -> tuple[bool, object]:
    try:
        return True, call.get(timeout=0)
    except TimeoutError:
        return False, None
    except Exception as e:
        return True, f"{type(e).__name__}: {e}"


@app.function(image=image, cpu=1, memory=1024, timeout=5 * HOURS, nonpreemptible=True)
def driver(spec: dict, manifest: dict) -> dict:
    """Rollout nodes first, the trainer once a rollout container runs, then watch and tear everything down."""
    run_id, roles = spec["run_id"], manifest["roles"]
    pool_token = secrets.token_urlsafe(24)
    rollout_fn = rollout.with_options(**_options(roles["rollout"]))
    calls = {f"rollout-{i}": rollout_fn.spawn(spec, pool_token, i) for i in (0, 1)}
    print(f"rollout calls: { {k: c.object_id for k, c in calls.items()} }", flush=True)
    state[f"{run_id}/calls"] = {k: c.object_id for k, c in calls.items()}

    while not any(state.get(f"{run_id}/rollout-{i}") for i in (0, 1)):
        for name, call in calls.items():
            done, value = _finished(call)
            if done:
                raise RuntimeError(f"{name} ended before any rollout container ran: {value}")
        time.sleep(5)
    trainer_call = trainer.with_options(**_options(roles["trainer"])).spawn(spec, pool_token)
    print(f"trainer call: {trainer_call.object_id}", flush=True)
    state[f"{run_id}/calls"] = {**state[f"{run_id}/calls"], "trainer": trainer_call.object_id}

    outcome = {}
    while True:
        done, value = _finished(trainer_call)
        if done:
            outcome = {"ended": "trainer", "value": value}
            break
        for name, call in calls.items():
            done, value = _finished(call)
            if done:
                outcome = {"ended": name, "value": value}
                break
        if outcome:
            break
        time.sleep(10)
    print(f"tearing down: {outcome}", flush=True)
    for call in [trainer_call, *calls.values()]:
        call.cancel(terminate_containers=True)
    state[f"{run_id}/outcome"] = outcome
    return outcome


@app.local_entrypoint()
def launch(run_id: str, num_rollout: int = 0):
    spec, manifest = make_spec.remote(num_rollout, run_id)
    print(json.dumps(manifest["roles"], indent=1))
    call = driver.spawn(spec, manifest)
    print(f"driver call {call.object_id}, app {app.app_id}")
    root = Path(os.environ["MILES_MONITOR_DIR"]) / "manifests"
    root.mkdir(parents=True, exist_ok=True)
    recipe = spec["recipe"]
    al = recipe["algorithm"]
    (root / f"{app.app_id}.json").write_text(
        json.dumps(
            {
                "app_id": app.app_id,
                "app_name": APP,
                "run_id": run_id,
                "kind": "train",
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "title": f"{recipe['name']}: {al['num_rollout']} steps x {al['rollout_batch_size']} tasks x {al['n_samples']} samples",
                "wandb_project": recipe["tracking"]["wandb_project"],
                "log_path": str(LOCAL / "logs" / f"{run_id}-follow.log"),
            },
            indent=1,
        )
    )
