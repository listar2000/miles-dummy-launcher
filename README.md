# miles-dummy-launcher

A throwaway launcher that runs one Miles RL run on the provider-agnostic image from outside the Miles repo, the way a
real launcher would: it reads the image's manifest, starts the node roles as subprocesses, wires them together and
watches the run. See `fork/guide/launcher.md` in listar2000/miles for the contract it follows.

- `launcher.py`: Modal. Rollout nodes, a trainer, a non-preemptible driver, keyless GCS for weight sync
  (workload identity federation with Modal's identity token), registration with the run viewer.
- `daytona_launch.py`: the rollout node on a Daytona GPU sandbox instead, trainer on Modal. Preflight probes included;
  blocked so far by the Daytona org's outbound network tier.
- `recipe.yaml`, `e2e_tasks/`: Qwen3-8B LoRA on DAPO math with a single-turn grader.
- `watch.py`, `follow.sh`: watchdog, and the log feed for the viewer.

Setup: copy `site.example.json` to `site.json` and fill in your GCP and W&B names; put `WANDB_API_KEY` (and
`DAYTONA_API_KEY` for Daytona) in `$MILES_DOTENV_DIR/.env`; write `prompts.jsonl` (DAPO rows, see the recipe).

    python -m modal run --detach launcher.py::launch --run-id <id> > logs/<id>.log 2>&1
