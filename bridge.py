"""Copy this run's state from the Modal Dict to the viewer's /ingest, which nodes would push to with MILES_STATE=http."""
import json, os, sys, time, urllib.parse, urllib.request
from pathlib import Path
import modal

run_id, url = sys.argv[1], sys.argv[2].rstrip("/")
token = Path.home().joinpath("miles-ui-data/ingest.token").read_text().strip()
d = modal.Dict.from_name("miles-rl-e2e-state")
while True:
    for key in d.keys():
        if key.endswith(run_id) and ":" in key:
            req = urllib.request.Request(
                f"{url}/{urllib.parse.quote(run_id, safe='')}/{urllib.parse.quote(key, safe='')}",
                data=json.dumps(d[key]).encode(), method="PUT",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            )
            try:
                urllib.request.urlopen(req, timeout=15).read()
            except Exception as e:
                print(key, type(e).__name__, e, flush=True)
    print(time.strftime("%H:%M:%S"), "pushed", flush=True)
    time.sleep(30)
