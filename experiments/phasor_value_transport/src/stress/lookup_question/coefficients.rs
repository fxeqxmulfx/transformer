//! Frozen magnitude ablations of the associative coefficient table.
//! PathCoefficients proves the margin decomposition and the cubic scaling.
use super::*;
use std::ops::Range;

pub(super) fn table_range(cfg: &Config) -> Result<Range<usize>> {
    if cfg.head.peer_read().and_then(|r| r.path).is_some_and(|p| p.agreement) {
        return Err(
            "a single-coordinate intervention requires a single-bank coefficient table".into(),
        );
    }
    trained_range(cfg)
}

pub(super) fn trained_range(cfg: &Config) -> Result<Range<usize>> {
    if cfg.head.peer_read().and_then(|r| r.path).is_some_and(|p| p.soft_address) {
        return Err("the scalar address read has no discrete relation coefficient table".into());
    }
    let layout = crate::slot_memory::Layout::new(cfg);
    let base = architecture::memory_start(cfg);
    let start = layout
        .path_start
        .ok_or("checkpoint has no learned lookup coefficients")?;
    Ok(base + start..base + layout.end() - layout.d - 1)
}

/// Selection always uses the original coefficients. Equal magnitudes at
/// the boundary are included together. Gain also uses the original maximum.
pub(super) fn transform(w: &mut [f32], selection: &str, gain: &str) -> Result<Value> {
    if w.is_empty() || !matches!(gain, "linear" | "cubic") {
        return Err("a nonempty table and a linear or cubic gain are required".into());
    }
    let mut ordered: Vec<_> = w.iter().map(|x| x.abs()).collect();
    ordered.sort_unstable_by(|a, b| b.total_cmp(a));
    let cutoff = if selection == "all" {
        None
    } else {
        let (mode, k) = selection
            .split_once(':')
            .ok_or("use all, keep:K or drop:K")?;
        let k: usize = k.parse()?;
        if !(1..=w.len()).contains(&k) || !matches!(mode, "keep" | "drop") {
            return Err("invalid lookup coefficient rank or selection".into());
        }
        Some((mode, ordered[k - 1]))
    };
    let floor = 1.0 / w.len() as f64;
    let maximum = ordered[0];
    let denominator = f64::from(maximum).powi(2) + floor;
    for v in w.iter_mut() {
        if cutoff.is_some_and(|(mode, t)| (v.abs() >= t) != (mode == "keep")) {
            *v = 0.;
        } else if gain == "cubic" {
            *v = crate::typed_lookup::scaled_weight(*v, floor, denominator, 1);
        }
    }
    Ok(
        json!({"selection":selection,"gain":gain,"threshold":cutoff.map(|(_,t)| t),
        "maximum":maximum,"floor":floor,"total_bins":w.len(),
        "kept_nonzero":w.iter().filter(|&&v| v!=0.).count()}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ties_are_included_together_and_cubic_preserves_the_leader() {
        let original = [-2., 1., -1., 0.25, 0.];
        let mut retained = original;
        let info = transform(&mut retained, "keep:2", "linear").unwrap();
        assert_eq!(retained, [-2., 1., -1., 0., 0.]);
        assert_eq!(info["kept_nonzero"], 3);
        let mut omitted = original;
        transform(&mut omitted, "drop:2", "linear").unwrap();
        for ((a, b), c) in retained.into_iter().zip(omitted).zip(original) {
            assert_eq!(a + b, c);
        }
        let mut scaled = original;
        transform(&mut scaled, "all", "cubic").unwrap();
        assert_eq!(scaled[0], original[0]);
        assert!(scaled[1] > 0. && scaled[1] < original[1]);
        assert_eq!(scaled[1], -scaled[2]);
        assert_eq!(scaled[4], 0.);
        for bad in ["keep:0", "drop:6", "keep", "remove:1"] {
            assert!(transform(&mut scaled, bad, "linear").is_err());
        }
        let mut extremes = [0., f32::MAX, -f32::MAX, 1.];
        transform(&mut extremes, "all", "cubic").unwrap();
        assert_eq!(extremes, [0., f32::MAX, -f32::MAX, 0.]);
    }
}
