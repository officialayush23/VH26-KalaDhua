#!/usr/bin/env bash
#
# Download open cache traces for the reuse head.
#
# None of these carry cost metadata. They train the *reuse* model only; the
# economic head (regen_cost_usd, cost_variance_ratio, log_regen_p50_ms) is
# trained on our own traces, where the cost vector is measured rather than
# guessed. See README.md, "Where the data comes from".
#
# Everything is opt-in per source, because these are large and several of them
# have terms you should read before mirroring them:
#
#   ./scripts/fetch_public_traces.sh twitter
#   ./scripts/fetch_public_traces.sh wiki
#   ./scripts/fetch_public_traces.sh libcachesim
#   ./scripts/fetch_public_traces.sh tencent
#   ./scripts/fetch_public_traces.sh list
#
set -euo pipefail

DEST="${AURA_TRAIN_TRACE_DIR:-data/traces/public}"
mkdir -p "$DEST"

have() { command -v "$1" >/dev/null 2>&1; }

require() {
  [[ "${DRY:-0}" == "1" ]] && return 0
  for tool in "$@"; do
    if ! have "$tool"; then
      echo "missing required tool: $tool" >&2
      exit 1
    fi
  done
}

note() { printf '\n== %s\n' "$*"; }

# ---------------------------------------------------------------------------

fetch_twitter() {
  note "Twitter production cache traces (Yang et al., OSDI '20)"
  cat <<'TXT'
54 anonymised Twitter in-memory cache clusters, one file per cluster, sorted by
time. Files are named cluster<N>.sort.zst and range from ~100 MB to ~30 GB
compressed. Columns are:

    timestamp,anonymized_key,key_size,value_size,client_id,operation,ttl

The whole set is ~2.8 TB; take one or two clusters. cluster17 and cluster44 are
commonly used because they are large enough to be interesting and small enough
to fit on a laptop.

Source and terms: https://github.com/twitter/cache-trace
Data is hosted on the project's S3 bucket; the repository README carries the
current URLs and the citation you are expected to include.
TXT
  require curl zstd
  local base="${AURA_TWITTER_BASE:-}"
  if [[ -z "$base" ]]; then
    echo "set AURA_TWITTER_BASE to the bucket URL from the cache-trace README, then re-run" >&2
    return 0
  fi
  for cluster in ${AURA_TWITTER_CLUSTERS:-17 44}; do
    local name="cluster${cluster}.sort.zst"
    echo "downloading $name"
    curl -fL --retry 3 -o "$DEST/$name" "$base/$name"
    zstd -d -f "$DEST/$name" -o "$DEST/cluster${cluster}.csv"
    echo "wrote $DEST/cluster${cluster}.csv"
  done
  cat <<TXT

Read it with:
  python -c "from pathlib import Path; from aura_train.traces import read_public_csv; \\
    print(sum(1 for _ in read_public_csv(Path('$DEST/cluster17.csv'), limit=1000)))"
TXT
}

fetch_wiki() {
  note "Wikimedia CDN request traces"
  cat <<'TXT'
Wikimedia publishes sampled CDN request logs (the "caching" analytics datasets)
under https://analytics.wikimedia.org/published/datasets/. The one most useful
here is the per-request upload/text cache sample: gzipped TSV with a relative
timestamp, a hashed URL and a response size, a few hundred MB per day.

They are sampled 1:1000 or 1:100 depending on the dataset, which matters: the
absolute request rate is not real, but reuse *structure* survives sampling, so
the reuse head trains fine and any rate-derived feature (freq_*, ewma_*) is
learned on a compressed timescale. Note that in the README when you report
numbers from it.

Terms: CC0, with the usual "do not attempt re-identification" clause.
TXT
  require curl
  local base="${AURA_WIKI_BASE:-https://analytics.wikimedia.org/published/datasets/caching}"
  echo "index: $base"
  echo "pick a dataset directory and a day, then:"
  echo "  curl -fL -o $DEST/wiki_sample.tsv.gz '$base/<dataset>/<file>.gz'"
}

fetch_libcachesim() {
  note "libCacheSim / cache-dataset oracleGeneral traces"
  cat <<'TXT'
The libCacheSim ecosystem republishes many public traces in one compact binary
format, oracleGeneral: fixed 24-byte records of

    uint32 real_time_seconds, uint64 obj_id, uint32 obj_size, int64 next_access_vtime

That last field is an oracle (Belady's future). Our reader deliberately ignores
it -- reading it would be textbook label leakage -- but it is exactly what you
want for computing a Belady upper bound in the benchmark, which is the engine's
job, not the model's.

The collection includes MSR Cambridge block traces, the Wikipedia CDN traces,
the Twitter clusters, Meta/Facebook KV traces and several CDN traces, all
converted to the same format. Sizes range from tens of MB to hundreds of GB.

Source: https://github.com/1a1a11a/libCacheSim (see doc/quickstart and the
cache-dataset links in the README for the current mirrors).
TXT
  require curl
  local base="${AURA_LIBCACHESIM_BASE:-}"
  if [[ -z "$base" ]]; then
    echo "set AURA_LIBCACHESIM_BASE to the mirror from the libCacheSim README, then re-run" >&2
    return 0
  fi
  for name in ${AURA_LIBCACHESIM_FILES:-}; do
    echo "downloading $name"
    curl -fL --retry 3 -o "$DEST/$name" "$base/$name"
  done
}

fetch_tencent() {
  note "Tencent / Alibaba block storage traces"
  cat <<'TXT'
Both are block-level, not object-level: records are (timestamp, volume, offset,
length, read/write). To use them as a cache trace you fix a block size and treat
(volume, offset / block_size) as the key. The reader does this when you give it
the offset column as the key column.

  Tencent: SNIA IOTTA repository, "Tencent Block Storage" -- 4,995 volumes,
           ~10 TB uncompressed across the full set; single-volume files are a
           few hundred MB.
  Alibaba: https://github.com/alibaba/block-traces -- 1,000 volumes, 30 days,
           ~440 GB compressed.

These are the right stress test for the scan and working-set-explosion regimes:
block workloads have long sequential sweeps that object caches rarely see.
Terms: SNIA IOTTA requires registration and citation; Alibaba's is Apache-2.0.
TXT
  echo "SNIA IOTTA index: http://iotta.snia.org/traces/parallel"
  echo "Alibaba:          https://github.com/alibaba/block-traces"
}

fetch_extra() {
  note "Other sources worth knowing about"
  cat <<'TXT'
  SEC EDGAR access logs
      https://www.sec.gov/about/data/edgar-log-file-data-sets
      Daily zipped CSVs of every EDGAR document fetch: ip (anonymised), date,
      time, cik, accession, extension, code, size, ... Roughly 200-400 MB per
      day. This one is unusual and useful: it has a *size* column and a stable
      document identity, and its access pattern is genuinely bursty around
      filing deadlines, which is a real-world flash crowd rather than a
      simulated one.

  IBM Cloud Object Storage traces
      http://iotta.snia.org/traces/key-value  (IBM COS, 98 buckets, 7 days)
      Object-level GET/PUT with object sizes. Closest public analogue to the
      workload AURA is designed for. No cost metadata, same caveat as the rest.

  MSR Cambridge block traces
      http://iotta.snia.org/traces/block-io  -- small, old, and the standard
      smoke test that everyone's cache simulator already agrees on. Useful for
      checking that our reader and our replay agree with published hit rates.
TXT
}

usage() {
  cat <<'TXT'
usage: fetch_public_traces.sh <source>

  twitter      Twitter production cache traces (cluster*.sort.zst)
  wiki         Wikimedia CDN request samples
  libcachesim  oracleGeneral binary traces (MSR, CDN, Twitter, Meta, ...)
  tencent      Tencent and Alibaba block traces
  extra        SEC EDGAR, IBM COS, MSR Cambridge
  list         print all of the above without downloading anything
TXT
}

case "${1:-list}" in
  twitter) fetch_twitter ;;
  wiki) fetch_wiki ;;
  libcachesim) fetch_libcachesim ;;
  tencent) fetch_tencent ;;
  extra) fetch_extra ;;
  list)
    DRY=1
    usage
    note "Destination: $DEST"
    fetch_twitter
    fetch_wiki
    fetch_libcachesim
    fetch_tencent
    fetch_extra
    ;;
  *) usage; exit 1 ;;
esac
