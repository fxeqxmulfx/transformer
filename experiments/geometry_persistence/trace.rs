//! DP1: apply the proved recurrence to every realized projected update.
use crate::{
    Result,
    classes::Partition,
    evidence::{hash, require},
    forecast_jet::norm,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
};
const E: usize = 97 * 128;
fn row(r: &mut BufReader<File>, h: &mut Sha256) -> Result<Vec<f64>> {
    let mut b = vec![0; 4 * E];
    r.read_exact(&mut b)?;
    h.update(&b);
    let v: Vec<_> = b
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x) as f64)
        .collect();
    require(v.iter().all(|x| x.is_finite()), "nonfinite trace")?;
    Ok(v)
}
fn hex(h: Sha256) -> String {
    h.finalize().iter().map(|x| format!("{x:02x}")).collect()
}
pub fn share(active: f64, unused: f64, between: f64) -> f64 {
    if active <= 0. {
        0.
    } else {
        active * active / (active * active + unused * unused + between * between)
    }
}
pub fn run(
    root: &Path,
    reference: &Value,
    active: &[bool],
    output: &Path,
    verify: bool,
) -> Result<Value> {
    let p = Partition::new(active.to_vec(), 128);
    let mut trace = BufReader::new(File::open(root.join("embeddings.bin"))?);
    let mut th = Sha256::new();
    let mut header = [0; 24];
    trace.read_exact(&mut header)?;
    th.update(header);
    require(
        header
            .as_chunks::<8>()
            .0
            .iter()
            .map(|x| u64::from_le_bytes(*x))
            .collect::<Vec<_>>()
            == [1, 10001, E as u64],
        "wrong trace shape",
    )?;
    let mut old = row(&mut trace, &mut th)?;
    let mut updates = BufReader::new(File::open(root.join("updates.jsonl"))?);
    let mut uh = Sha256::new();
    let mut line = String::new();
    let mut writer = if verify {
        None
    } else {
        Some(BufWriter::new(File::create_new(output)?))
    };
    let mut reader = if verify {
        Some(BufReader::new(File::open(output)?))
    } else {
        None
    };
    let mut anchor: [Vec<f64>; 3] = std::array::from_fn(|_| vec![]);
    let mut anchor_norm = [0.; 3];
    let mut radii = [0.; 3];
    let mut global_radius = 0.;
    let mut q = 1.;
    let mut prefixes = 0;
    let mut moving = 0;
    let mut minimum = [1f64; 3];
    let mut thresholds = [[0usize; 4]; 2];
    let mut max_excess = 0f64;
    let mut max_defect = 0f64;
    let mut worst_radius = 0.;
    let mut worst_budget = 0.;
    let mut max_distance = 0f64;
    let mut path_length = 0.;
    let mut anchor_embedding = vec![];
    for step in 1..=10000 {
        line.clear();
        require(updates.read_line(&mut line)? > 0, "missing update")?;
        uh.update(line.as_bytes());
        let r: Value = serde_json::from_str(&line)?;
        require(r["step"] == step, "wrong update order")?;
        let eta = r["eta"].as_f64().ok_or("missing eta")?;
        let a = 1. - 0.1 * eta;
        require((0.0..=1.0).contains(&a), "invalid decay")?;
        let new = row(&mut trace, &mut th)?;
        if step == 4000 {
            anchor = p.project(&new);
            anchor_norm = anchor.each_ref().map(|x| norm(x));
            worst_radius = new.iter().map(|x| x.abs()).fold(0., f64::max);
            anchor_embedding = new.clone();
        }
        if (4001..=8000).contains(&step) {
            let before = p.project(&old);
            let after = p.project(&new);
            let forcing: Vec<_> = new.iter().zip(&old).map(|(n, o)| n - a * o).collect();
            let projected = p.project(&forcing);
            q *= a;
            global_radius = (a * global_radius + crate::matrix::norm_upper(&forcing)).next_up();
            for j in 0..3 {
                let defect = norm(
                    &(0..E)
                        .map(|k| after[j][k] - a * before[j][k] - projected[j][k])
                        .collect::<Vec<_>>(),
                );
                max_defect = max_defect.max(defect);
                radii[j] =
                    (a * radii[j] + crate::matrix::norm_upper(&projected[j]) + defect).next_up();
                let distance = norm(
                    &(0..E)
                        .map(|k| after[j][k] - q * anchor[j][k])
                        .collect::<Vec<_>>(),
                );
                max_excess =
                    max_excess.max((distance - radii[j]) / (1. + distance.abs().max(radii[j])));
                let energy = norm(&after[j]).powi(2);
                let recorded = r["after"][j].as_f64().ok_or("missing energy")?;
                require(
                    (energy - recorded).abs() <= 1e-8 * (1. + energy.abs()),
                    "projection differs from source",
                )?;
            }
            let lower = (q * anchor_norm[0] - radii[0]).max(0.);
            let upper = [q * anchor_norm[1] + radii[1], q * anchor_norm[2] + radii[2]];
            let bound = share(lower, upper[0], upper[1]);
            let common = share(
                (q * anchor_norm[0] - global_radius).max(0.),
                q * anchor_norm[1].hypot(anchor_norm[2]) + global_radius,
                0.,
            );
            let actual_norm = after.each_ref().map(|x| norm(x));
            let actual = share(actual_norm[0], actual_norm[1], actual_norm[2]);
            for b in [bound, common] {
                max_excess = max_excess.max(b - actual);
            }
            minimum[0] = minimum[0].min(bound);
            minimum[1] = minimum[1].min(common);
            minimum[2] = minimum[2].min(actual);
            for (j, b) in [bound, common].into_iter().enumerate() {
                for (i, t) in [0., 0.5, 0.6, 2. / 3.].into_iter().enumerate() {
                    thresholds[j][i] += usize::from(b > t + 1e-8);
                }
            }
            let distance = norm(
                &new.iter()
                    .zip(&anchor_embedding)
                    .map(|(n, o)| n - o)
                    .collect::<Vec<_>>(),
            );
            max_distance = max_distance.max(distance);
            let displacement = norm(&new.iter().zip(&old).map(|(n, o)| n - o).collect::<Vec<_>>());
            moving += usize::from(displacement > 0.);
            path_length += displacement;
            let rounding = r["rounding_max"].as_f64().ok_or("missing rounding")?;
            let direction =
                (19f64 / 14.).sqrt() * (1. - 0.95f64.powi(step)).sqrt() / (1. - 0.9f64.powi(step));
            worst_budget += eta * (0.1 * worst_radius + direction) + rounding;
            worst_radius = a * worst_radius + eta * direction + rounding;
            let record = json!([step, q, radii, global_radius, bound, common, actual]);
            if let Some(w) = &mut writer {
                writeln!(w, "{record}")?;
            }
            if let Some(r) = &mut reader {
                let mut saved = String::new();
                require(r.read_line(&mut saved)? > 0, "missing saved prefix")?;
                require(
                    serde_json::from_str::<Value>(&saved)? == record,
                    "prefix evidence differs",
                )?;
            }
            prefixes += 1;
        }
        old = new;
    }
    require(
        trace.read(&mut [0])? == 0 && updates.read_line(&mut String::new())? == 0,
        "extra source rows",
    )?;
    if let Some(mut w) = writer {
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    if let Some(mut r) = reader {
        require(r.read_line(&mut String::new())? == 0, "extra saved prefix")?;
    }
    let trace_hash = hex(th);
    let updates_hash = hex(uh);
    require(
        reference["trace_sha256"] == trace_hash && reference["updates_sha256"] == updates_hash,
        "source trace changed",
    )?;
    require(
        prefixes == 4000 && max_excess <= 1e-8,
        "prefix bound failed",
    )?;
    Ok(
        json!({"status":"pass","updates":10000,"prefixes":prefixes,"moving_updates":moving,
        "anchor_norms":anchor_norm,"final_decay":q,"final_projected_radii":radii,"final_global_radius":global_radius,
        "minimum_projected_share_bound":minimum[0],"minimum_global_share_bound":minimum[1],"minimum_actual_share":minimum[2],
        "thresholds":[0.,0.5,0.6,2./3.],"certified_prefixes":thresholds,"max_scaled_bound_excess":max_excess,
        "max_projection_arithmetic_defect":max_defect,"previous_worst_case_motion_bound":(E as f64).sqrt()*worst_budget,
        "actual_max_displacement":max_distance,"actual_path_length":path_length,
        "trace_sha256":trace_hash,"updates_sha256":updates_hash,"prefix_sha256":hash(&fs::read(output)?)}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn forcing_bound_preserves_scale_but_accounts_for_competing_motion() {
        assert_eq!(share(3., 4., 0.), 9. / 25.);
        assert_eq!(share(6., 8., 0.), 9. / 25.);
        assert!(share(2.5, 4.5, 0.) < share(3., 4., 0.));
        assert_eq!(share(0., 2., 1.), 0.);
        let p = Partition::new(vec![true, true, false, false], 1);
        let x = [1., 3., -2., 2.];
        let f = [0.2, -0.2, 0.1, -0.1];
        let next: Vec<_> = x.iter().zip(f).map(|(x, f)| 0.9 * x + f).collect();
        let a = p.project(&x);
        let b = p.project(&next);
        let pf = p.project(&f);
        for j in 0..3 {
            let d: Vec<_> = b[j].iter().zip(&a[j]).map(|(b, a)| b - 0.9 * a).collect();
            assert!((norm(&d) - norm(&pf[j])).abs() < 1e-12);
        }
    }
}
