//! CL1-CL4: independent CE calculation and exact potential accounting.
use crate::{
    Result,
    classes::{Partition, dot, energy},
    inputs::require,
};
use serde_json::{Value, json};

pub fn potential(w: &[f64], m: &[f64], s: f64) -> f64 {
    w.iter()
        .zip(m)
        .map(|(w, m)| (w - 9. * s * m).powi(2) + 90. * s * s * m * m)
        .sum()
}
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 * (1. + a.abs().max(b.abs()))
}
pub struct Batch {
    pub gradient: Vec<f64>,
    pub q: f64,
    pub mean_feature_energy: f64,
    pub maximum_feature_energy: f64,
    pub loss: f64,
}
pub fn ce(features: &[f64], labels: &[usize], e: &[f64], part: &Partition) -> Result<Batch> {
    let width = part.width;
    let classes = part.active.len();
    require(
        !labels.is_empty() && features.len() == labels.len() * width,
        "batch shape",
    )?;
    let mut gradient = vec![0.; e.len()];
    let (mut q, mut total, mut max_energy, mut loss) = (0f64, 0., 0f64, 0.);
    for (x, &y) in features.chunks_exact(width).zip(labels) {
        require(y < classes && part.active[y], "unused training label")?;
        let logits: Vec<_> = e.chunks_exact(width).map(|row| dot(row, x)).collect();
        let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = logits.iter().map(|z| (z - top).exp()).sum();
        loss += sum.ln() + top - logits[y];
        let f = energy(x);
        total += f;
        max_energy = max_energy.max(f);
        for c in 0..classes {
            let p = (logits[c] - top).exp() / sum;
            if !part.active[c] {
                q = q.max(p);
            }
            let d = (p - f64::from(c == y)) / labels.len() as f64;
            for k in 0..width {
                gradient[c * width + k] += d * x[k];
            }
        }
    }
    Ok(Batch {
        gradient,
        q,
        mean_feature_energy: total / labels.len() as f64,
        maximum_feature_energy: max_energy,
        loss: loss / labels.len() as f64,
    })
}
pub struct Step<'a> {
    pub old_p: &'a [f64],
    pub old_m: &'a [f64],
    pub new_p: &'a [f64],
    pub new_m: &'a [f64],
    pub new_v: &'a [f64],
    pub readout: &'a [f64],
    pub encoder: &'a [f64],
    pub eta: f64,
    pub step: usize,
    pub l: f64,
}
pub fn diagnose(part: &Partition, a: &Step<'_>) -> Result<Value> {
    let project = |v: &[f64]| part.project(v)[1].clone();
    let (w, m, wn, mn, g) = (
        project(a.old_p),
        project(a.old_m),
        project(a.new_p),
        project(a.new_m),
        project(a.readout),
    );
    let enc = project(a.encoder);
    let b1 = 1. - 0.9f64.powi(a.step as i32);
    let b2 = (1. - 0.95f64.powi(a.step as i32)).sqrt();
    let diagonal: Vec<_> = a
        .new_v
        .iter()
        .map(|v| 1. / (v.sqrt() / b2 + 1e-8))
        .collect();
    let alpha = diagonal.iter().copied().fold(f64::INFINITY, f64::min);
    let raw = a.eta * alpha / b1;
    let safe = if a.l > 0. {
        raw.min(1. / (19. * a.l))
    } else {
        raw
    };
    let radial = dot(&w, &g);
    let g2 = energy(&g);
    require(
        g2 <= a.l * radial + 1e-8 * (1. + g2.abs().max((a.l * radial).abs())),
        "CE cocoercivity violation",
    )?;
    require(
        radial >= -1e-8 * (1. + energy(&w).sqrt() * g2.sqrt()),
        "CE damping violation",
    )?;
    let mr: Vec<_> = m.iter().zip(&g).map(|(m, g)| 0.9 * m + 0.1 * g).collect();
    let source_direction = project(
        &diagonal
            .iter()
            .zip(a.new_m)
            .map(|(d, m)| d * m)
            .collect::<Vec<_>>(),
    );
    let mut tracks = vec![];
    for (name, s) in [("raw", raw), ("stable_control", safe)] {
        let wr: Vec<_> = w.iter().zip(&mr).map(|(w, m)| w - s * m).collect();
        let mut dw = vec![vec![0.; w.len()]; 4];
        let mut dm = vec![vec![0.; w.len()]; 4];
        for k in 0..w.len() {
            dw[0][k] = -a.eta * 0.1 * w[k];
            dw[1][k] = s * mn[k] - a.eta / b1 * source_direction[k];
            dm[2][k] = 0.1 * enc[k];
            dw[2][k] = -s * dm[2][k];
            dm[3][k] = mn[k] - mr[k] - dm[2][k];
            let parameter_round =
                wn[k] - (1. - a.eta * 0.1) * w[k] + a.eta / b1 * source_direction[k];
            dw[3][k] = parameter_round - s * dm[3][k];
            require(
                close(wn[k] - wr[k], dw.iter().map(|x| x[k]).sum()),
                "position perturbation closure",
            )?;
            require(
                close(mn[k] - mr[k], dm.iter().map(|x| x[k]).sum()),
                "memory perturbation closure",
            )?;
        }
        let old = potential(&w, &m, s);
        let reference = potential(&wr, &mr, s);
        let actual = potential(&wn, &mn, s);
        let baseline_terms = [
            -2. * s * radial,
            19. * s * s * g2,
            -17.1 * s * s * energy(&m.iter().zip(&g).map(|(m, g)| m - g).collect::<Vec<_>>()),
        ];
        let baseline = baseline_terms.iter().sum::<f64>();
        require(
            close(reference - old, baseline),
            "momentum potential identity",
        )?;
        let linear: Vec<_> = (0..4)
            .map(|j| {
                (0..w.len())
                    .map(|k| {
                        2. * (wr[k] - 9. * s * mr[k]) * (dw[j][k] - 9. * s * dm[j][k])
                            + 180. * s * s * mr[k] * dm[j][k]
                    })
                    .sum::<f64>()
            })
            .collect();
        let quadratic = (0..w.len())
            .map(|k| {
                let x = dw.iter().map(|x| x[k]).sum::<f64>();
                let y = dm.iter().map(|x| x[k]).sum::<f64>();
                (x - 9. * s * y).powi(2) + 90. * s * s * y * y
            })
            .sum::<f64>();
        let change = actual - old;
        let closure = (change - baseline - linear.iter().sum::<f64>() - quadratic).abs()
            / (1. + old.abs().max(actual.abs()));
        require(closure <= 1e-8, "full potential identity")?;
        let stable = 19. * s * a.l <= 2.;
        if stable {
            require(
                baseline <= 1e-8 * (1. + old),
                "stable reference increased potential",
            )?;
        }
        tracks.push(json!({"method":name,"s":s,"stability_number":19.*s*a.l,
            "sufficient_condition":stable,"before":old,"reference_after":reference,
            "actual_after":actual,"baseline_terms":baseline_terms,"baseline_change":baseline,
            "actual_change":change,"actual_increase":change>1e-8*(1.+old.abs().max(actual.abs())),
            "perturbation_linear":linear,"perturbation_quadratic":quadratic,
            "closure_scaled":closure}));
    }
    Ok(json!({"alpha":alpha,"L":a.l,"unused_gradient_energy":g2,
        "unused_radial":radial,"unused_parameter_energy":energy(&w),
        "unused_moment_energy":energy(&m),"terms":["decay","adaptive","tied_encoder","execution"],
        "tracks":tracks}))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_ce_cocoercivity_and_nonzero_memory_identity() {
        let p = Partition::new(vec![true, true, false, false], 2);
        let w = [1., 0.5, -2., 1., 3., -1., -1., 2.];
        let x = [2., -1., -0.5, 3.];
        let batch = ce(&x, &[0, 1], &w, &p).unwrap();
        let g = p.project(&batch.gradient)[1].clone();
        let e = p.project(&w)[1].clone();
        let l = batch.q * batch.mean_feature_energy;
        assert!(energy(&g) > 0. && energy(&g) <= l * dot(&e, &g) + 1e-12);
        let m: Vec<_> = e
            .iter()
            .enumerate()
            .map(|(k, v)| v * 0.3 + (k as f64).sin())
            .collect();
        let s = 1. / (19. * l);
        let mr: Vec<_> = m.iter().zip(&g).map(|(m, g)| 0.9 * m + 0.1 * g).collect();
        let wr: Vec<_> = e.iter().zip(&mr).map(|(w, m)| w - s * m).collect();
        let delta = -2. * s * dot(&e, &g) + 19. * s * s * energy(&g)
            - 17.1 * s * s * energy(&m.iter().zip(&g).map(|(m, g)| m - g).collect::<Vec<_>>());
        assert!(delta < 0. && close(potential(&wr, &mr, s) - potential(&e, &m, s), delta));
    }
    #[test]
    fn partial_gradient_matches_central_difference() {
        let p = Partition::new(vec![true, true, false, false], 2);
        let w = [1., 0.5, -2., 1., 3., -1., -1., 2.];
        let x = [2., -1., -0.5, 3.];
        let g = ce(&x, &[0, 1], &w, &p).unwrap().gradient;
        for k in 0..8 {
            let (mut plus, mut minus) = (w, w);
            plus[k] += 1e-5;
            minus[k] -= 1e-5;
            let d = (ce(&x, &[0, 1], &plus, &p).unwrap().loss
                - ce(&x, &[0, 1], &minus, &p).unwrap().loss)
                / 2e-5;
            assert!((d - g[k]).abs() < 1e-8);
        }
    }
}
