#!/usr/bin/env python3
"""Push trained bundles to Supabase and (optionally) make them active.

This is the last step of a training run and the only one that changes what the
production engine serves, so it verifies before it uploads: a bundle that fails
``verify_bundle.py`` is never published.

Usage::

    export SUPABASE_URL=https://<project>.supabase.co
    export SUPABASE_SERVICE_ROLE_SECRET_KEY=<service role key>

    python scripts/push_model.py models/reuse_gbdt_h60s.json --activate
    python scripts/push_model.py models/*.json                 # upload, stay inactive
    python scripts/push_model.py models/reuse_gbdt_h60s.json --activate --reload-url \\
        http://localhost:8080
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aura_train.export import read_bundle  # noqa: E402
from aura_train.supabase_io import (  # noqa: E402
    DEFAULT_MODEL_BUCKET,
    Session,
    push_event,
    register_model,
    set_active,
    upload_bundle,
)
from scripts.verify_bundle import verify  # noqa: E402

LOG = logging.getLogger("push_model")


def reload_engine(base_url: str) -> None:
    """Ask a running engine to pick the new active bundle up."""
    import requests

    url = f"{base_url.rstrip('/')}/v1/model/reload"
    response = requests.post(url, json={"source": "supabase"}, timeout=30)
    response.raise_for_status()
    LOG.info("engine reload: %s", response.text.strip()[:200])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="publish AURA model bundles")
    parser.add_argument("bundles", nargs="+", type=Path)
    parser.add_argument("--bucket", default=DEFAULT_MODEL_BUCKET)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--activate", action="store_true", help="flip is_active for each name")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--reload-url", help="engine base URL to POST /v1/model/reload to")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if not args.skip_verify:
        for bundle_path in args.bundles:
            problems = verify(bundle_path, args.dataset_dir, 1000, 42)
            if problems:
                print(f"refusing to publish {bundle_path}:")
                for problem in problems:
                    print(f"  {problem}")
                return 1

    session = Session()
    published: list[tuple[str, str, str]] = []
    for bundle_path in args.bundles:
        bundle = read_bundle(bundle_path)
        onnx_path = bundle_path.with_suffix(".onnx")
        storage_path, onnx_storage = upload_bundle(
            bundle_path,
            onnx_path if onnx_path.exists() else None,
            bucket=args.bucket,
            session=session,
        )
        register_model(
            bundle_path,
            storage_path,
            onnx_storage,
            is_active=args.activate,
            session=session,
        )
        if args.activate:
            set_active(str(bundle["name"]), str(bundle["version"]), session=session)
        published.append((str(bundle["name"]), str(bundle["version"]), storage_path))
        print(f"pushed {bundle['name']} {bundle['version']} -> {storage_path}")

    try:
        push_event(
            "ModelPublished",
            {
                "models": [{"name": n, "version": v, "path": p} for n, v, p in published],
                "activated": bool(args.activate),
            },
            session=session,
        )
    except Exception as exc:  # noqa: BLE001 - the event log must never fail a publish
        LOG.warning("could not write aura_events row: %s", exc)

    if args.reload_url:
        reload_engine(args.reload_url)
    else:
        print("")
        print("tell the running engine to load it:")
        print(
            '  curl -X POST http://localhost:8080/v1/model/reload '
            '-H "Content-Type: application/json" -d \'{"source":"supabase"}\''
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
