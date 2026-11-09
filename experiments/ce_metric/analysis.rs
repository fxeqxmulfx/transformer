//! FM1-FM3: weighted CE and the complete signed full-update identity.
use crate::{
    Result,
    classes::{Partition, dot, energy},
    evidence::require,
    reference,
    state::State,
};
use serde_json::{Value, json};
pub struct Executed {
    pub features: Vec<f64>,
    pub partial: Vec<f64>,
    pub full: Vec<f64>,
    pub rows: Vec<u32>,
    pub norm: f64,
}
pub fn potential(w: &[f64], m: &[f64], r: &[f64], s: f64) -> f64 {
    w.iter()
        .zip(m)
        .enumerate()
        .map(|(k, (w, m))| {
            let r = r[k % r.len()];
            (w - 9. * s * r * m).powi(2) / r + 90. * s * s * r * m * m
        })
        .sum()
}
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 * (1. + a.abs().max(b.abs()))
}
fn weighted(x: &[f64], r: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(k, x)| r[k % r.len()] * x * x)
        .sum()
}
pub fn assess(old: &State, new: &State, f: &Value, a: &Executed) -> Result<Value> {
    let active: Vec<bool> = serde_json::from_value(f["active"].clone())?;
    let part = Partition::new(active, 128);
    require(
        a.rows.len().is_multiple_of(3)
            && a.partial.len() == old.p.len()
            && a.full.len() == old.p.len()
            && a.norm.is_finite()
            && a.norm >= 0.,
        "executed array shape/norm",
    )?;
    let labels: Vec<_> = a
        .rows
        .as_chunks::<3>()
        .0
        .iter()
        .map(|r| r[2] as usize)
        .collect();
    let batch = reference::ce(&a.features, &labels, &old.p, &part)?;
    require(
        batch.maximum_feature_energy <= 128. * (1. + 5e-6),
        "feature normalization",
    )?;
    let error = batch
        .gradient
        .iter()
        .zip(&a.partial)
        .map(|(x, y)| (x - y).abs() / (1. + x.abs().max(y.abs())))
        .fold(0., f64::max);
    require(error <= 1e-5, "independent CE gradient")?;
    let clip = 1. / a.norm.max(1.);
    let project = |v: &[f64]| part.project(v)[1].clone();
    let (w, m, wn, mn) = (
        project(&old.p),
        project(&old.m),
        project(&new.p),
        project(&new.m),
    );
    let g = project(&batch.gradient.iter().map(|g| clip * g).collect::<Vec<_>>());
    let enc = project(
        &a.full
            .iter()
            .zip(&a.partial)
            .map(|(f, r)| clip * (f - r))
            .collect::<Vec<_>>(),
    );
    let anchor = f["anchor"].as_u64().ok_or("anchor")? as i32;
    let b1 = 1. - 0.9f64.powi(anchor + 1);
    let b2 = (1. - 0.95f64.powi(anchor + 1)).sqrt();
    let eta = f["eta"].as_f64().ok_or("eta")?;
    let raw = f["s"].as_f64().ok_or("s")?;
    let diagonal: Vec<_> = new.v.iter().map(|v| 1. / (v.sqrt() / b2 + 1e-8)).collect();
    let direction = project(
        &diagonal
            .iter()
            .zip(&new.m)
            .map(|(d, m)| d * m)
            .collect::<Vec<_>>(),
    );
    let radial = dot(&w, &g);
    require(
        radial >= -1e-8 * (1. + energy(&w).sqrt() * energy(&g).sqrt()),
        "CE radial damping",
    )?;
    let mr: Vec<_> = m.iter().zip(&g).map(|(m, g)| 0.9 * m + 0.1 * g).collect();
    let feature_r: Vec<f64> = serde_json::from_value(f["r"].clone())?;
    require(
        feature_r.len() == 128 && feature_r.iter().all(|x| x.is_finite() && *x >= 1.),
        "feature metric",
    )?;
    let mut tracks = vec![];
    for (metric, r) in [("isotropic", vec![1.; 128]), ("feature", feature_r)] {
        let rmax = r.iter().copied().fold(0., f64::max);
        let mean = a
            .features
            .as_chunks::<128>()
            .0
            .iter()
            .map(|x| weighted(x, &r))
            .sum::<f64>()
            / labels.len() as f64;
        let l = clip * batch.q * mean;
        let global = 128. * (1. + 5e-6) * rmax;
        let g2 = weighted(&g, &r);
        require(
            g2 <= l * radial + 1e-8 * (1. + g2.abs().max((l * radial).abs())),
            "weighted CE inequality",
        )?;
        let safe = if l > 0. { raw.min(1. / (19. * l)) } else { raw };
        for (method, s) in [("raw", raw), ("stable_control", safe)] {
            let wr: Vec<_> = w
                .iter()
                .zip(&mr)
                .enumerate()
                .map(|(k, (w, m))| w - s * r[k % 128] * m)
                .collect();
            let mut dw = vec![vec![0.; w.len()]; 4];
            let mut dm = dw.clone();
            for k in 0..w.len() {
                let h = s * r[k % 128];
                dw[0][k] = -eta * 0.1 * w[k];
                dw[1][k] = h * mn[k] - eta / b1 * direction[k];
                dm[2][k] = 0.1 * enc[k];
                dw[2][k] = -h * dm[2][k];
                dm[3][k] = mn[k] - mr[k] - dm[2][k];
                let ep = wn[k] - (1. - eta * 0.1) * w[k] + eta / b1 * direction[k];
                dw[3][k] = ep - h * dm[3][k];
                require(
                    close(wn[k] - wr[k], dw.iter().map(|x| x[k]).sum())
                        && close(mn[k] - mr[k], dm.iter().map(|x| x[k]).sum()),
                    "source decomposition",
                )?;
            }
            let before = potential(&w, &m, &r, s);
            let reference = potential(&wr, &mr, &r, s);
            let actual = potential(&wn, &mn, &r, s);
            let baseline_terms = [
                -2. * s * radial,
                19. * s * s * g2,
                -17.1
                    * s
                    * s
                    * weighted(
                        &m.iter().zip(&g).map(|(m, g)| m - g).collect::<Vec<_>>(),
                        &r,
                    ),
            ];
            let baseline = baseline_terms.iter().sum::<f64>();
            require(
                close(reference - before, baseline),
                "weighted reference identity",
            )?;
            let linear: Vec<_> = (0..4)
                .map(|j| {
                    (0..w.len())
                        .map(|k| {
                            let r = r[k % 128];
                            2. * (wr[k] - 9. * s * r * mr[k]) * (dw[j][k] - 9. * s * r * dm[j][k])
                                / r
                                + 180. * s * s * r * mr[k] * dm[j][k]
                        })
                        .sum::<f64>()
                })
                .collect();
            let tw: Vec<_> = (0..w.len())
                .map(|k| dw.iter().map(|x| x[k]).sum())
                .collect();
            let tm: Vec<_> = (0..w.len())
                .map(|k| dm.iter().map(|x| x[k]).sum())
                .collect();
            let quadratic = potential(&tw, &tm, &r, s);
            let delta = actual - before;
            let scale = 1. + before.abs().max(actual.abs());
            let closure = (delta - baseline - linear.iter().sum::<f64>() - quadratic).abs() / scale;
            require(closure <= 1e-8, "weighted full potential identity")?;
            let stable = 19. * s * l <= 2.;
            if stable {
                require(
                    baseline <= 1e-8 * (1. + before),
                    "weighted stable reference increase",
                )?;
            }
            require(
                energy(&w) <= 1.9 * rmax * before + 1e-8 * (1. + energy(&w)),
                "ordinary energy conversion",
            )?;
            tracks.push(json!({"metric":metric,"method":method,"s":s,"rmax":rmax,"L":l,
                "global_L":global,"stability_number":19.*s*l,"sufficient_condition":stable,
                "global_condition_raw":19.*raw*global<=2.,"global_stability_number_raw":19.*raw*global,
                "before":before,"reference_after":reference,"actual_after":actual,
                "baseline_terms":baseline_terms,"baseline_change":baseline,"actual_change":delta,
                "actual_increase":delta>1e-8*scale,"linear":linear,"quadratic":quadratic,
                "relative_actual_change":delta/before,"relative_baseline_change":baseline/before,
                "relative_linear":linear.iter().map(|x|x/before).collect::<Vec<_>>(),
                "relative_quadratic":quadratic/before,"closure_scaled":closure,
                "ordinary_energy_bound":1.9*rmax*before,"ordinary_energy_bound_ratio":1.9*rmax*before/energy(&w)}));
        }
    }
    Ok(
        json!({"tracks":tracks,"q":batch.q,"clip":clip,"eta":eta,"partial_gradient_error":error,
        "maximum_feature_energy":batch.maximum_feature_energy,"unused_radial":radial,
        "unused_parameter_energy":energy(&w),"terms":["decay","adaptive_remainder","tied_encoder","execution"]}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weighted_actual_ce_and_arbitrary_memory() {
        let p = Partition::new(vec![true, true, false, false], 2);
        let w = [1., 0.5, -2., 1., 3., -1., -1., 2.];
        let x = [2., -1., -0.5, 3.];
        let r = [1., 3.];
        let b = reference::ce(&x, &[0, 1], &w, &p).unwrap();
        let g = p.project(&b.gradient)[1].clone();
        let w = p.project(&w)[1].clone();
        let l = b.q
            * x.as_chunks::<2>()
                .0
                .iter()
                .map(|x| weighted(x, &r))
                .sum::<f64>()
            / 2.;
        assert!(weighted(&g, &r) <= l * dot(&w, &g) + 1e-12);
        let m: Vec<_> = w
            .iter()
            .enumerate()
            .map(|(k, w)| 0.3 * w + (k as f64).sin())
            .collect();
        let s = 1. / (19. * l);
        let mn: Vec<_> = m.iter().zip(&g).map(|(m, g)| 0.9 * m + 0.1 * g).collect();
        let wn: Vec<_> = w
            .iter()
            .zip(&mn)
            .enumerate()
            .map(|(k, (w, m))| w - s * r[k % 2] * m)
            .collect();
        let delta = -2. * s * dot(&w, &g) + 19. * s * s * weighted(&g, &r)
            - 17.1
                * s
                * s
                * weighted(
                    &m.iter().zip(&g).map(|(m, g)| m - g).collect::<Vec<_>>(),
                    &r,
                );
        assert!(delta < 0. && close(potential(&wn, &mn, &r, s) - potential(&w, &m, &r, s), delta));
        assert!(energy(&w) <= 1.9 * 3. * potential(&w, &m, &r, s));
        assert!(close(
            potential(&w, &m, &[1., 1.], s),
            reference::potential(&w, &m, s)
        ));
    }
}
