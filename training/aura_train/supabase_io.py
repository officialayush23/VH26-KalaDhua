"""Supabase I/O: model bundles, traces and benchmark results.

Credentials are read from the environment and never from a file in the repo:

* ``SUPABASE_URL``                        project URL
* ``SUPABASE_SERVICE_ROLE_SECRET_KEY``    service role key (server-side only)
* ``SUPABASE_DIRECT_CONNECTION_URL``      Postgres URL, used only by ``sql/``

The ``supabase`` Python client is the intended transport. It is not always
installable in the environments where this pipeline runs, so there is a REST
fallback over ``requests`` that speaks the same PostgREST and Storage APIs. Both
paths produce identical rows and identical objects; the SDK is preferred when
present because it keeps up with API changes for us.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

LOG = logging.getLogger(__name__)

MODELS_TABLE = "aura_models"
RUNS_TABLE = "aura_benchmark_runs"
RESULTS_TABLE = "aura_benchmark_results"
TRACES_TABLE = "aura_traces"
EVENTS_TABLE = "aura_events"

DEFAULT_MODEL_BUCKET = "aura-models"
DEFAULT_TRACE_BUCKET = "aura-traces"


class SupabaseConfigError(RuntimeError):
    """Raised when the environment does not carry usable credentials."""


class SupabaseRequestError(RuntimeError):
    """Raised when Supabase rejects a request."""


@dataclass(frozen=True)
class Credentials:
    url: str
    service_key: str

    @property
    def rest_url(self) -> str:
        return f"{self.url.rstrip('/')}/rest/v1"

    @property
    def storage_url(self) -> str:
        return f"{self.url.rstrip('/')}/storage/v1"


def credentials(require: bool = True) -> Credentials | None:
    """Read credentials from the environment."""
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_SECRET_KEY", "").strip()
    if not url or not key:
        if require:
            raise SupabaseConfigError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_SECRET_KEY must be set. "
                "In Colab put them in the secrets panel; locally export them, "
                "never commit them."
            )
        return None
    return Credentials(url=url, service_key=key)


def direct_connection_url() -> str | None:
    """Postgres connection string, used to apply ``sql/*.sql``."""
    value = os.environ.get("SUPABASE_DIRECT_CONNECTION_URL", "").strip()
    return value or None


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


def _sdk_client(creds: Credentials) -> Any | None:
    try:
        from supabase import create_client
    except ImportError:
        return None
    try:
        return create_client(creds.url, creds.service_key)
    except Exception as exc:  # pragma: no cover - network/credential dependent
        LOG.warning("supabase client construction failed (%s); using REST fallback", exc)
        return None


class Session:
    """One connection's worth of state. Cheap to construct, safe to reuse."""

    def __init__(self, creds: Credentials | None = None, prefer_sdk: bool = True) -> None:
        self.creds = creds or credentials()
        if self.creds is None:  # pragma: no cover - credentials() raises by default
            raise SupabaseConfigError("no Supabase credentials")
        self.sdk = _sdk_client(self.creds) if prefer_sdk else None
        if self.sdk is None:
            LOG.info("using the REST fallback transport")

    # -- REST helpers ----------------------------------------------------

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "apikey": self.creds.service_key,
            "Authorization": f"Bearer {self.creds.service_key}",
        }

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        import requests

        headers = dict(self._headers)
        headers.update(kwargs.pop("headers", {}))
        response = requests.request(method, url, headers=headers, timeout=120, **kwargs)
        if response.status_code >= 400:
            raise SupabaseRequestError(
                f"{method} {url} -> {response.status_code}: {response.text[:500]}"
            )
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return response.content

    # -- storage ---------------------------------------------------------

    def ensure_bucket(self, bucket: str, public: bool = False) -> None:
        """Create the bucket if it does not exist. Idempotent."""
        if self.sdk is not None:
            try:
                self.sdk.storage.get_bucket(bucket)
                return
            except Exception:  # noqa: BLE001 - SDK raises many bucket-missing types
                LOG.info("creating storage bucket %s", bucket)
                try:
                    self.sdk.storage.create_bucket(bucket, options={"public": public})
                    return
                except Exception as exc:  # noqa: BLE001
                    if "already exists" in str(exc).lower():
                        return
                    raise
        try:
            self._request("GET", f"{self.creds.storage_url}/bucket/{bucket}")
            return
        except SupabaseRequestError:
            LOG.info("creating storage bucket %s", bucket)
            try:
                self._request(
                    "POST",
                    f"{self.creds.storage_url}/bucket",
                    json={"id": bucket, "name": bucket, "public": public},
                )
            except SupabaseRequestError as exc:
                if "already exists" not in str(exc).lower():
                    raise

    def storage_upload(
        self,
        bucket: str,
        object_path: str,
        data: bytes,
        content_type: str | None = None,
        upsert: bool = True,
    ) -> str:
        self.ensure_bucket(bucket)
        mime = content_type or mimetypes.guess_type(object_path)[0] or "application/octet-stream"
        if self.sdk is not None:
            self.sdk.storage.from_(bucket).upload(
                path=object_path,
                file=data,
                file_options={"content-type": mime, "upsert": "true" if upsert else "false"},
            )
            return f"{bucket}/{object_path}"
        self._request(
            "POST",
            f"{self.creds.storage_url}/object/{bucket}/{object_path}",
            data=data,
            headers={"Content-Type": mime, "x-upsert": "true" if upsert else "false"},
        )
        return f"{bucket}/{object_path}"

    def storage_download(self, bucket: str, object_path: str) -> bytes:
        if self.sdk is not None:
            return bytes(self.sdk.storage.from_(bucket).download(object_path))
        payload = self._request("GET", f"{self.creds.storage_url}/object/{bucket}/{object_path}")
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
        return json.dumps(payload).encode("utf-8")

    # -- tables ----------------------------------------------------------

    def insert(self, table: str, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        if self.sdk is not None:
            response = self.sdk.table(table).insert(list(rows)).execute()
            return list(response.data or [])
        payload = self._request(
            "POST",
            f"{self.creds.rest_url}/{table}",
            json=list(rows),
            headers={"Content-Type": "application/json", "Prefer": "return=representation"},
        )
        return list(payload or [])

    def upsert(
        self, table: str, rows: Sequence[dict[str, Any]], on_conflict: str
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        if self.sdk is not None:
            response = self.sdk.table(table).upsert(list(rows), on_conflict=on_conflict).execute()
            return list(response.data or [])
        payload = self._request(
            "POST",
            f"{self.creds.rest_url}/{table}?on_conflict={on_conflict}",
            json=list(rows),
            headers={
                "Content-Type": "application/json",
                "Prefer": "return=representation,resolution=merge-duplicates",
            },
        )
        return list(payload or [])

    def update(self, table: str, filters: dict[str, Any], values: dict[str, Any]) -> None:
        if self.sdk is not None:
            query = self.sdk.table(table).update(values)
            for column, value in filters.items():
                query = query.eq(column, value)
            query.execute()
            return
        params = {column: f"eq.{value}" for column, value in filters.items()}
        self._request(
            "PATCH",
            f"{self.creds.rest_url}/{table}",
            params=params,
            json=values,
            headers={"Content-Type": "application/json", "Prefer": "return=minimal"},
        )

    def select(
        self,
        table: str,
        filters: dict[str, Any] | None = None,
        order: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if self.sdk is not None:
            query = self.sdk.table(table).select("*")
            for column, value in (filters or {}).items():
                query = query.eq(column, value)
            if order:
                query = query.order(order, desc=True)
            if limit:
                query = query.limit(limit)
            return list(query.execute().data or [])
        params: dict[str, Any] = {"select": "*"}
        for column, value in (filters or {}).items():
            params[column] = f"eq.{value}"
        if order:
            params["order"] = f"{order}.desc"
        if limit:
            params["limit"] = limit
        return list(self._request("GET", f"{self.creds.rest_url}/{table}", params=params) or [])


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def storage_path_for(name: str, version: str) -> str:
    """``reuse_gbdt_h60s/2026-09-04T11-20-00Z/model_bundle.json`` (contract 5.1)."""
    from .export import storage_version

    return f"{name}/{storage_version(version)}/model_bundle.json"


def upload_bundle(
    bundle_path: Path,
    onnx_path: Path | None = None,
    bucket: str = DEFAULT_MODEL_BUCKET,
    session: Session | None = None,
) -> tuple[str, str | None]:
    """Upload a bundle (and its ONNX sibling) to Storage.

    Returns ``(storage_path, onnx_path)`` as ``bucket/key`` strings, which is
    what goes into ``aura_models.storage_path``.
    """
    session = session or Session()
    bundle = json.loads(Path(bundle_path).read_text())
    key = storage_path_for(str(bundle["name"]), str(bundle["version"]))
    stored = session.storage_upload(
        bucket, key, Path(bundle_path).read_bytes(), "application/json"
    )
    stored_onnx: str | None = None
    if onnx_path is not None and Path(onnx_path).exists():
        onnx_key = key.rsplit("/", 1)[0] + "/model.onnx"
        stored_onnx = session.storage_upload(
            bucket, onnx_key, Path(onnx_path).read_bytes(), "application/octet-stream"
        )
    LOG.info("uploaded %s", stored)
    return stored, stored_onnx


def register_model(
    bundle_path: Path,
    storage_path: str,
    onnx_path: str | None = None,
    is_active: bool = False,
    session: Session | None = None,
) -> dict[str, Any]:
    """Insert the ``aura_models`` row that describes an uploaded bundle."""
    session = session or Session()
    bundle = json.loads(Path(bundle_path).read_text())
    row = {
        "id": str(uuid.uuid4()),
        "name": bundle["name"],
        "kind": bundle["kind"],
        "horizon_ms": int(bundle["horizon_ms"]),
        "version": bundle["version"],
        "storage_path": storage_path,
        "onnx_path": onnx_path,
        "metrics": bundle.get("metrics", {}),
        "feature_names": bundle.get("feature_names", []),
        "is_active": bool(is_active),
    }
    inserted = session.insert(MODELS_TABLE, [row])
    LOG.info("registered model %s version %s", row["name"], row["version"])
    return inserted[0] if inserted else row


def set_active(
    name: str,
    version: str,
    session: Session | None = None,
) -> None:
    """Make one version the active one for its model name.

    Deactivate first, then activate: the engine reads whichever row is active,
    so two active rows for one name is the one state we must never pass through.
    """
    session = session or Session()
    session.update(MODELS_TABLE, {"name": name}, {"is_active": False})
    session.update(MODELS_TABLE, {"name": name, "version": version}, {"is_active": True})
    LOG.info("model %s version %s is now active", name, version)


def list_models(
    name: str | None = None,
    limit: int = 50,
    session: Session | None = None,
) -> list[dict[str, Any]]:
    """Newest models first, optionally filtered by name."""
    session = session or Session()
    filters = {"name": name} if name else None
    return session.select(MODELS_TABLE, filters=filters, order="created_at", limit=limit)


def download_active_bundle(
    name: str,
    dest: Path | None = None,
    bucket: str = DEFAULT_MODEL_BUCKET,
    session: Session | None = None,
) -> dict[str, Any]:
    """Fetch the bundle currently marked active. This is the engine's boot path."""
    session = session or Session()
    rows = session.select(MODELS_TABLE, filters={"name": name, "is_active": True}, limit=1)
    if not rows:
        raise SupabaseRequestError(f"no active model named {name!r}")
    storage_path = str(rows[0]["storage_path"])
    bucket_name, _, key = storage_path.partition("/")
    if bucket_name != bucket:
        LOG.debug("storage_path names bucket %s", bucket_name)
    payload = session.storage_download(bucket_name or bucket, key)
    bundle = json.loads(payload.decode("utf-8"))
    if dest is not None:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(payload)
        LOG.info("wrote %s", dest)
    return bundle


def upload_trace(
    trace_path: Path,
    bucket: str = DEFAULT_TRACE_BUCKET,
    session: Session | None = None,
) -> dict[str, Any]:
    """Upload a trace and register it in ``aura_traces``."""
    from .traces import TraceMeta, meta_path_for

    session = session or Session()
    trace_path = Path(trace_path)
    meta = TraceMeta.from_path(trace_path)
    key = f"{meta.scenario}/{trace_path.name}"
    stored = session.storage_upload(bucket, key, trace_path.read_bytes(), "application/gzip")
    meta_file = meta_path_for(trace_path)
    if meta_file.exists():
        session.storage_upload(
            bucket, f"{key}.meta.json", meta_file.read_bytes(), "application/json"
        )
    row = {
        "id": str(uuid.uuid4()),
        "name": trace_path.name,
        "scenario": meta.scenario,
        "storage_path": stored,
        "rows": int(meta.requests),
        "unique_keys": int(meta.unique_keys),
        "bytes": int(trace_path.stat().st_size),
        "meta": {
            "seed": meta.seed,
            "duration_s": meta.duration_s,
            "generator_version": meta.generator_version,
            "applications": list(meta.applications),
        },
    }
    inserted = session.insert(TRACES_TABLE, [row])
    return inserted[0] if inserted else row


def push_benchmark_run(
    report: dict[str, Any],
    engine_version: str = "unknown",
    session: Session | None = None,
) -> dict[str, Any]:
    """Insert (or update) the ``aura_benchmark_runs`` row for a BenchmarkReport."""
    session = session or Session()
    row = {
        "id": str(uuid.uuid4()),
        "run_id": str(report["run_id"]),
        "scenario": report.get("scenario", "unknown"),
        "seed": int(report.get("seed", 0)),
        "capacity_bytes": int(report.get("capacity_bytes", 0)),
        "requests": int(report.get("requests", 0)),
        "engine_version": engine_version,
        "summary": {
            "winner": report.get("winner"),
            "improvement_vs": report.get("improvement_vs", {}),
            "belady_upper_bound": report.get("belady_upper_bound", {}),
        },
    }
    inserted = session.upsert(RUNS_TABLE, [row], on_conflict="run_id")
    return inserted[0] if inserted else row


def push_benchmark_results(
    run_id: str,
    rows: Iterable[dict[str, Any]],
    session: Session | None = None,
) -> int:
    """Insert one ``aura_benchmark_results`` row per policy."""
    session = session or Session()
    known = {
        "policy",
        "object_hit_rate",
        "byte_hit_rate",
        "p95_latency_ms",
        "backend_requests",
        "total_cost_usd",
        "regen_cost_usd",
        "sla_penalty_usd",
    }
    payload: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {"run_id": run_id}
        extra: dict[str, Any] = {}
        for key, value in row.items():
            if key in known:
                record[key] = value
            else:
                extra[key] = value
        record["decision_overhead_us"] = row.get("decision_overhead_us_p50")
        record["extra"] = extra
        payload.append(record)
    session.insert(RESULTS_TABLE, payload)
    LOG.info("pushed %d benchmark result rows for run %s", len(payload), run_id)
    return len(payload)


def push_event(kind: str, detail: dict[str, Any], session: Session | None = None) -> None:
    """Append to ``aura_events``. Used by the CLI to record training runs."""
    session = session or Session()
    session.insert(EVENTS_TABLE, [{"kind": kind, "detail": detail}])
