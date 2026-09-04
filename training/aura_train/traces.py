"""Trace readers.

Primary format is AURA's own gzipped CSV (contract section 4), written by
``aura-bench --emit-trace`` and ``aura sim --emit-trace``. Public research
traces are also supported so the reuse head can be trained on workloads nobody
in this project generated; see ``README.md`` for what each source is good for.

All readers yield :class:`~aura_train.features.AccessEvent`, so everything
downstream is format-agnostic.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import lzma
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterable, Iterator

from .features import AccessEvent

LOG = logging.getLogger(__name__)

TRACE_COLUMNS: tuple[str, ...] = (
    "ts_ms",
    "key_id",
    "application",
    "object_type",
    "size_bytes",
    "ttl_ms",
    "sla_class",
    "cpu_ms",
    "gpu_ms",
    "db_ms",
    "network_bytes",
    "api_calls",
    "api_cost_usd",
    "regen_latency_ms",
    "scenario",
    "regime",
)


@dataclass(frozen=True)
class TraceMeta:
    """Contents of the companion ``*.meta.json``."""

    scenario: str = "unknown"
    seed: int = 0
    requests: int = 0
    unique_keys: int = 0
    duration_s: float = 0.0
    generator_version: int = 1
    applications: tuple[str, ...] = ()

    @classmethod
    def from_path(cls, trace_path: Path) -> "TraceMeta":
        meta_path = meta_path_for(trace_path)
        if not meta_path.exists():
            LOG.warning("no meta file for %s, using defaults", trace_path)
            return cls()
        raw = json.loads(meta_path.read_text())
        return cls(
            scenario=str(raw.get("scenario", "unknown")),
            seed=int(raw.get("seed", 0)),
            requests=int(raw.get("requests", 0)),
            unique_keys=int(raw.get("unique_keys", 0)),
            duration_s=float(raw.get("duration_s", 0.0)),
            generator_version=int(raw.get("generator_version", 1)),
            applications=tuple(raw.get("applications", []) or []),
        )


def meta_path_for(trace_path: Path) -> Path:
    """``foo.csv.gz -> foo.meta.json``."""
    name = trace_path.name
    for suffix in (".csv.gz", ".csv", ".gz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return trace_path.with_name(f"{name}.meta.json")


def _open_text(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    if path.suffix in (".xz", ".lzma"):
        return lzma.open(path, "rt", newline="")
    return path.open("rt", newline="")


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# --------------------------------------------------------------------------
# AURA's own format
# --------------------------------------------------------------------------


def read_aura_trace(path: Path, limit: int | None = None) -> Iterator[AccessEvent]:
    """Read a contract-section-4 CSV(.gz) trace."""
    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in TRACE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: trace is missing columns {missing}")
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                return
            yield AccessEvent(
                ts_ms=_f(row, "ts_ms"),
                key_id=int(float(row["key_id"])),
                application=row["application"],
                object_type=row["object_type"],
                size_bytes=int(_f(row, "size_bytes")),
                ttl_ms=_f(row, "ttl_ms"),
                sla_class=row["sla_class"] or "normal",
                cpu_ms=_f(row, "cpu_ms"),
                gpu_ms=_f(row, "gpu_ms"),
                db_ms=_f(row, "db_ms"),
                network_bytes=_f(row, "network_bytes"),
                api_calls=_f(row, "api_calls"),
                api_cost_usd=_f(row, "api_cost_usd"),
                regen_latency_ms=_f(row, "regen_latency_ms"),
                scenario=row.get("scenario", "") or "unknown",
                regime=row.get("regime", "") or "unknown",
            )


def write_aura_trace(path: Path, events: Iterable[AccessEvent], meta: TraceMeta) -> int:
    """Write a trace plus its companion meta file. Used by the generator."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    unique: set[int] = set()
    first_ts = 0.0
    last_ts = 0.0
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRACE_COLUMNS)
        for event in events:
            if count == 0:
                first_ts = event.ts_ms
            last_ts = event.ts_ms
            unique.add(event.key_id)
            writer.writerow(
                [
                    f"{event.ts_ms:.0f}",
                    event.key_id,
                    event.application,
                    event.object_type,
                    event.size_bytes,
                    f"{event.ttl_ms:.0f}",
                    event.sla_class,
                    f"{event.cpu_ms:.4f}",
                    f"{event.gpu_ms:.4f}",
                    f"{event.db_ms:.4f}",
                    f"{event.network_bytes:.0f}",
                    f"{event.api_calls:.0f}",
                    f"{event.api_cost_usd:.8f}",
                    f"{event.regen_latency_ms:.4f}",
                    event.scenario,
                    event.regime,
                ]
            )
            count += 1
    payload = {
        "scenario": meta.scenario,
        "seed": meta.seed,
        "requests": count,
        "unique_keys": len(unique),
        "duration_s": max(last_ts - first_ts, 0.0) / 1000.0,
        "generator_version": meta.generator_version,
        "applications": sorted({"recommendation", "analytics", "content"} & set(meta.applications))
        or list(meta.applications),
    }
    meta_path_for(path).write_text(json.dumps(payload, indent=2) + "\n")
    return count


# --------------------------------------------------------------------------
# Public traces
# --------------------------------------------------------------------------

# libCacheSim "oracleGeneral" binary record: little-endian
#   uint32 real_time (seconds), uint64 obj_id, uint32 obj_size, int64 next_access_vtime
_ORACLE_GENERAL_STRUCT = struct.Struct("<IQIq")
ORACLE_GENERAL_RECORD_BYTES = _ORACLE_GENERAL_STRUCT.size  # 24


def read_oracle_general(
    path: Path,
    application: str = "public",
    limit: int | None = None,
    default_ttl_ms: float = 300_000.0,
) -> Iterator[AccessEvent]:
    """Read a libCacheSim ``oracleGeneral`` binary trace.

    These traces carry timestamps, object ids and object sizes and nothing else.
    There is no cost metadata, so every cost field is zero: such a trace can
    train the *reuse* head only. ``next_access_vtime`` is present in the format
    but deliberately ignored -- it is an oracle, and using it would be leakage.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "rb") as handle:  # type: ignore[operator]
        while True:
            chunk = handle.read(ORACLE_GENERAL_RECORD_BYTES * 4096)
            if not chunk:
                return
            for offset in range(0, len(chunk) - ORACLE_GENERAL_RECORD_BYTES + 1,
                                ORACLE_GENERAL_RECORD_BYTES):
                real_time, obj_id, obj_size, _next_vtime = _ORACLE_GENERAL_STRUCT.unpack_from(
                    chunk, offset
                )
                if limit is not None and count >= limit:
                    return
                count += 1
                yield AccessEvent(
                    ts_ms=float(real_time) * 1000.0,
                    key_id=int(obj_id),
                    application=application,
                    object_type="object",
                    size_bytes=int(obj_size),
                    ttl_ms=default_ttl_ms,
                    sla_class="normal",
                    cpu_ms=0.0,
                    gpu_ms=0.0,
                    db_ms=0.0,
                    network_bytes=float(obj_size),
                    api_calls=0.0,
                    api_cost_usd=0.0,
                    regen_latency_ms=0.0,
                    scenario=path.stem,
                    regime="public",
                )


# Column aliases seen in the public CSV traces we support. Everything is
# normalised onto (timestamp, key, size).
_CSV_TIME_KEYS = ("timestamp", "time", "ts", "real_time", "reqtime", "unix_time")
_CSV_KEY_KEYS = (
    "anonymized_key",
    "key",
    "obj_id",
    "object_id",
    "id",
    "hash",
    "uri",
    "url",
    "offset",
    "blockid",
)
_CSV_SIZE_KEYS = (
    "value_size",
    "size",
    "obj_size",
    "object_size",
    "bytes",
    "response_size",
    "content_length",
)


def _pick(fieldnames: Iterable[str], candidates: Iterable[str]) -> str | None:
    lowered = {name.lower().strip(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def read_public_csv(
    path: Path,
    application: str = "public",
    limit: int | None = None,
    time_unit: str = "s",
    default_size_bytes: int = 4096,
    default_ttl_ms: float = 300_000.0,
    delimiter: str | None = None,
) -> Iterator[AccessEvent]:
    """Read a Wikipedia CDN / Twitter cluster / Tencent-style CSV trace.

    These formats differ in column naming but agree on the three things we need:
    a timestamp, an object identifier and a size. Headerless files are accepted
    and interpreted positionally as ``time, key, size``.

    ``time_unit`` is ``"s"``, ``"ms"`` or ``"us"``.

    Like the binary public traces, these carry no cost metadata -- reuse head
    only.
    """
    scale = {"s": 1000.0, "ms": 1.0, "us": 0.001}[time_unit]
    with _open_text(path) as handle:
        sample = handle.read(8192)
        handle.seek(0)
        if delimiter is None:
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",\t; ").delimiter
            except csv.Error:
                delimiter = ","
        has_header = any(c.isalpha() for c in sample.splitlines()[0]) if sample else False
        keyspace: dict[str, int] = {}

        def key_to_id(raw: str) -> int:
            try:
                return int(raw)
            except ValueError:
                got = keyspace.get(raw)
                if got is None:
                    got = len(keyspace)
                    keyspace[raw] = got
                return got

        rows: Iterator[dict[str, str]]
        if has_header:
            dict_reader = csv.DictReader(handle, delimiter=delimiter)
            names = dict_reader.fieldnames or []
            t_col = _pick(names, _CSV_TIME_KEYS) or names[0]
            k_col = _pick(names, _CSV_KEY_KEYS) or names[min(1, len(names) - 1)]
            s_col = _pick(names, _CSV_SIZE_KEYS)
            rows = iter(dict_reader)
            for i, row in enumerate(rows):
                if limit is not None and i >= limit:
                    return
                size = int(_f(row, s_col, default_size_bytes)) if s_col else default_size_bytes
                yield _public_event(
                    float(_f(row, t_col)) * scale,
                    key_to_id(str(row.get(k_col, i))),
                    size,
                    application,
                    default_ttl_ms,
                    path.stem,
                )
        else:
            plain = csv.reader(handle, delimiter=delimiter)
            for i, cells in enumerate(plain):
                if limit is not None and i >= limit:
                    return
                if len(cells) < 2:
                    continue
                size = int(float(cells[2])) if len(cells) > 2 and cells[2] else default_size_bytes
                yield _public_event(
                    float(cells[0]) * scale,
                    key_to_id(cells[1]),
                    size,
                    application,
                    default_ttl_ms,
                    path.stem,
                )


def _public_event(
    ts_ms: float,
    key_id: int,
    size_bytes: int,
    application: str,
    ttl_ms: float,
    scenario: str,
) -> AccessEvent:
    return AccessEvent(
        ts_ms=ts_ms,
        key_id=key_id,
        application=application,
        object_type="object",
        size_bytes=max(size_bytes, 0),
        ttl_ms=ttl_ms,
        sla_class="normal",
        cpu_ms=0.0,
        gpu_ms=0.0,
        db_ms=0.0,
        network_bytes=float(max(size_bytes, 0)),
        api_calls=0.0,
        api_cost_usd=0.0,
        regen_latency_ms=0.0,
        scenario=scenario,
        regime="public",
    )


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

READERS = ("aura", "oracle_general", "public_csv")


def detect_format(path: Path) -> str:
    """Guess the trace format from the filename and, for CSVs, the header."""
    name = path.name.lower()
    if name.endswith(".oracleGeneral".lower()) or name.endswith(".oraclegeneral.bin"):
        return "oracle_general"
    if name.endswith(".bin") or name.endswith(".oraclegeneral.gz"):
        return "oracle_general"
    if name.endswith(".csv") or name.endswith(".csv.gz") or name.endswith(".csv.xz"):
        try:
            with _open_text(path) as handle:
                header = handle.readline()
        except OSError:
            return "public_csv"
        return "aura" if "ts_ms" in header and "key_id" in header else "public_csv"
    return "public_csv"


def read_trace(
    path: Path,
    fmt: str | None = None,
    limit: int | None = None,
    application: str = "public",
) -> Iterator[AccessEvent]:
    """Read any supported trace, auto-detecting the format when not given."""
    resolved = fmt or detect_format(path)
    LOG.info("reading %s as %s", path, resolved)
    if resolved == "aura":
        return read_aura_trace(path, limit=limit)
    if resolved == "oracle_general":
        return read_oracle_general(path, application=application, limit=limit)
    if resolved == "public_csv":
        return read_public_csv(path, application=application, limit=limit)
    raise ValueError(f"unknown trace format {resolved!r}, expected one of {READERS}")


def discover_traces(trace_dir: Path) -> list[Path]:
    """All readable traces under ``trace_dir``, sorted for reproducibility."""
    patterns = ("*.csv.gz", "*.csv", "*.bin", "*.oracleGeneral", "*.oracleGeneral.bin")
    found: list[Path] = []
    for pattern in patterns:
        found.extend(sorted(trace_dir.rglob(pattern)))
    return sorted(set(found))


def sniff_header(path: Path, n_bytes: int = 512) -> str:
    """Small helper used by the CLI's ``inspect`` output."""
    with _open_text(path) as handle:
        return io.StringIO(handle.read(n_bytes)).getvalue()
