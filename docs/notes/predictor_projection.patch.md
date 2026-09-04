# Patch: score a bundle by feature *name*, not by position

Apply to `engine/aura-server/src/predictor.rs`.

## Why

`score()` currently feeds the engine's whole 16-element feature array straight into the
model, so tree split `feature[7]` means "whatever the engine happens to keep at index 7".
That is fine while the bundle was trained on exactly the same 16 columns in exactly the same
order, and silently catastrophic the moment it was not — no error, no warning, just
confident nonsense.

Decision D8 moves the model to 12 portable features (dropping `regen_cost_usd`,
`log_regen_p50_ms`, `cost_variance_ratio` and `app_id`, which do not transfer between
applications). The engine still computes all 16 — the economic layer needs the cost ones —
so the bundle has to say which columns it wants and the loader has to honour that.

This is also the general fix. Once a bundle is projected by name, the engine and the trainer
can evolve their feature sets independently, and a mismatch becomes a startup error instead
of a wrong answer.

## 1. Add a projection field to the bundle struct

`ModelBundle` gains one runtime-only field (not serialised, so existing bundles still load):

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBundle {
    // ... existing fields unchanged ...
    pub metrics: serde_json::Value,

    /// Engine feature index for each of the bundle's own columns, resolved at load time.
    /// `None` until `resolve_projection` has run.
    #[serde(skip)]
    pub projection: Option<Vec<usize>>,
}
```

## 2. Resolve it when the bundle is loaded

Add this free function:

```rust
/// Map each of the bundle's declared feature names onto the engine's feature index.
///
/// Returns an error listing every unknown name rather than silently dropping columns: a
/// bundle that asks for a feature this engine does not compute is a version mismatch, and
/// the only safe thing to do is refuse it.
fn resolve_projection(names: &[String]) -> anyhow::Result<Vec<usize>> {
    use aura_core::features::FEATURE_NAMES;

    let mut projection = Vec::with_capacity(names.len());
    let mut unknown = Vec::new();
    for name in names {
        match FEATURE_NAMES.iter().position(|f| f == name) {
            Some(idx) => projection.push(idx),
            None => unknown.push(name.clone()),
        }
    }
    if !unknown.is_empty() {
        anyhow::bail!(
            "model bundle needs features this engine does not compute: {}. \
             Engine features are: {}",
            unknown.join(", "),
            FEATURE_NAMES.join(", ")
        );
    }
    Ok(projection)
}
```

And call it in `load_bundle`, before the bundle is stored:

```rust
pub fn load_bundle(&mut self, mut bundle: ModelBundle, source: &str) {
    match resolve_projection(&bundle.feature_names) {
        Ok(projection) => {
            tracing::info!(
                name = %bundle.name,
                features = bundle.feature_names.len(),
                %source,
                "loaded model bundle"
            );
            bundle.projection = Some(projection);
        }
        Err(err) => {
            // Refusing to load leaves the previous predictor in place, which is the
            // conservative outcome: a stale model beats a mis-indexed one.
            tracing::error!(name = %bundle.name, %source, %err, "rejected model bundle");
            return;
        }
    }
    // ... existing body, unchanged ...
}
```

## 3. Project before normalising

Replace `normalize` with a version that gathers the bundle's own columns first:

```rust
fn normalize(b: &ModelBundle, f: &Features) -> Vec<f64> {
    // Gather the columns this bundle was actually trained on. Without the projection a
    // 12-feature model would read the engine's first 12 columns, which are not the same
    // twelve.
    let picked: Vec<f64> = match &b.projection {
        Some(p) => p.iter().map(|&i| f.get(i).copied().unwrap_or(0.0)).collect(),
        None => f.to_vec(),
    };

    match &b.normalization {
        Some(n) => picked
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let m = n.mean.get(i).copied().unwrap_or(0.0);
                let s = n.scale.get(i).copied().unwrap_or(1.0);
                if s.abs() < 1e-12 {
                    0.0
                } else {
                    (v - m) / s
                }
            })
            .collect(),
        None => picked,
    }
}
```

`score()` needs no change: it already consumes whatever `normalize` returns.

## 4. Test

Add to `predictor.rs`:

```rust
#[cfg(test)]
mod projection_tests {
    use super::*;

    #[test]
    fn a_bundle_may_use_a_subset_of_the_engine_features() {
        let names: Vec<String> = ["trend", "freq_1m", "log_size_bytes"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let p = resolve_projection(&names).expect("all three are engine features");
        use aura_core::features::idx;
        assert_eq!(p, vec![idx::TREND, idx::FREQ_1M, idx::LOG_SIZE_BYTES]);
    }

    #[test]
    fn an_unknown_feature_is_rejected_rather_than_ignored() {
        let names = vec!["trend".to_string(), "phase_of_the_moon".to_string()];
        let err = resolve_projection(&names).unwrap_err().to_string();
        assert!(err.contains("phase_of_the_moon"), "error should name the missing feature");
    }

    #[test]
    fn projection_reorders_correctly() {
        // Deliberately not in engine order: the bundle's order is what counts.
        let names: Vec<String> = ["log_size_bytes", "log_age_ms"].iter().map(|s| s.to_string()).collect();
        let p = resolve_projection(&names).unwrap();
        let mut f = [0.0f64; aura_core::features::N_FEATURES];
        f[aura_core::features::idx::LOG_AGE_MS] = 1.5;
        f[aura_core::features::idx::LOG_SIZE_BYTES] = 9.5;
        let picked: Vec<f64> = p.iter().map(|&i| f[i]).collect();
        assert_eq!(picked, vec![9.5, 1.5]);
    }
}
```

Then:

```powershell
cargo test -p aura-server predictor
```
