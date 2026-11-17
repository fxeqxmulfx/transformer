//! Evaluate CM116's one-class strict-margin inequalities on the original
//! training rows of each frozen arithmetic endpoint. The focal class has
//! the most training misses, with ties resolved by the smaller object ID.
use baseline::{channel::argmax, hash};
use bilinear::guard;
use composed_retrieval::{Result, data};
use learned_retrieval::cohort::{raw_file, read};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path};

fn interval(raw: &[u8], labels: &[usize], v: usize, bound: f64) -> Result<Value> {
    if v < 2
        || labels.is_empty()
        || raw.len() != 8 * v * labels.len()
        || labels.iter().any(|&y| y >= v)
        || !(bound > 0. && bound.is_finite())
    {
        return Err("invalid training logits, targets or offset box".into());
    }
    let mut misses = vec![0usize; v];
    let rows = || {
        raw.chunks_exact(8 * v).map(|row| {
            row.as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b))
                .collect::<Vec<_>>()
        })
    };
    for (z, &y) in rows().zip(labels) {
        if z.iter().any(|a| !a.is_finite()) {
            return Err("nonfinite training logit".into());
        }
        misses[y] += usize::from(argmax(&z) != y);
    }
    let wrong = misses.iter().sum::<usize>();
    let Some(k) = (0..v)
        .filter(|&k| misses[k] > 0)
        .max_by_key(|&k| (misses[k], std::cmp::Reverse(k)))
    else {
        return Ok(json!({"targets":labels.len(),"wrong":0,"focal_class":null}));
    };
    let (mut lower, mut upper): (Option<f64>, Option<f64>) = (None, None);
    let mut other_violations = 0;
    let mut positives = 0;
    for (z, &y) in rows().zip(labels) {
        if k == y {
            positives += 1;
            let rival = (0..v)
                .filter(|&c| c != k)
                .map(|c| z[c])
                .fold(f64::NEG_INFINITY, f64::max);
            let delta = rival - z[k];
            lower = Some(lower.map_or(delta, |a| a.max(delta)));
        } else {
            let delta = z[y] - z[k];
            upper = Some(upper.map_or(delta, |a| a.min(delta)));
            other_violations += usize::from((0..v).any(|c| c != k && c != y && z[c] >= z[y]));
        }
    }
    let feasible = other_violations == 0
        && lower.zip(upper).is_none_or(|(l, u)| l < u)
        && lower.is_none_or(|l| l < bound)
        && upper.is_none_or(|u| -bound < u);
    Ok(
        json!({"targets":labels.len(),"wrong":wrong,"focal_class":k,"focal_targets":positives,
        "focal_misses":misses[k],"strict_lower":lower,"strict_upper":upper,
        "other_rival_violations":other_violations,"box_feasible":feasible}),
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [work, storage, reference] = args.as_slice() else {
        return Err("expected WORKSPACE STORAGE LOGODDS_ARITHMETIC".into());
    };
    let reference = Path::new(reference);
    guard::admit(1 << 30, Path::new(work), Path::new(storage))?;
    let root = reference.join("original/run");
    let seal = read(&root.join("sealed/manifest.json"))?;
    let identity = &seal["identity"];
    if identity["protocol_sha256"] != hash(data::JOINT_PROTOCOL.as_bytes())
        || identity["binary_sha256"]
            != hash(&fs::read(reference.join("original/composed-retrieval"))?)
        || identity["producer"] != fs::read_to_string(reference.join("producer.commit"))?.trim()
    {
        return Err("unexpected arithmetic reference".into());
    }
    let p = data::protocol()?;
    let bound = p.output_offset_bound.ok_or("offset box")?;
    let mut cases = BTreeMap::new();
    for (id, spec) in p
        .cases
        .iter()
        .filter(|(_, s)| s.source_fingerprints.is_some())
    {
        let (task, pins) = data::original(&p, id)?;
        let labels: Vec<_> = task.splits["train"]
            .iter()
            .flat_map(|e| &e.targets)
            .filter(|&&y| y >= 0)
            .map(|&y| y as usize)
            .collect();
        let dir = root.join(id);
        let report = read(&dir.join("scores/report.json"))?;
        let mut cells = BTreeMap::new();
        for mode in &p.modes {
            for seed in &p.seeds {
                let cell = format!("{mode}-{seed}");
                let key = format!("{cell}-final");
                let bytes = fs::read(dir.join(&cell).join("receipt.json"))?;
                let receipt: Value = serde_json::from_slice(&bytes)?;
                let raw = fs::read(raw_file(&dir.join("scores"), &key, "train"))?;
                let train = &report["cells"][key]["splits"]["train"];
                let result = interval(&raw, &labels, spec.vocabulary, bound)?;
                if receipt["identity"] != *identity
                    || receipt["data_fingerprints"] != pins
                    || seal["cells"][format!("{id}/{cell}")] != hash(&bytes)
                    || train["raw_sha256"] != hash(&raw)
                    || train["metrics"]["targets"] != labels.len()
                    || result["wrong"].as_u64().ok_or("misses")?
                        + train["metrics"]["correct"]
                            .as_u64()
                            .ok_or("training correct")?
                        != labels.len() as u64
                {
                    return Err(format!("training reference mismatch for {id}/{cell}").into());
                }
                cells.insert(cell, result);
            }
        }
        cases.insert(id, cells);
    }
    if cases.len() != 48 {
        return Err("incomplete arithmetic training inventory".into());
    }
    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &json!({"box":bound,"cases":cases}),
    )?;
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn raw(z: &[f64]) -> Vec<u8> {
        z.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn strict_bounds_include_the_two_flat_spectrum_examples() -> Result<()> {
        for (a, b, feasible) in [
            (1., 3., true),
            (2., 1., false),
            (2., 2., false),
            (16., 17., false),
        ] {
            let r = interval(&raw(&[0., a, 0., b]), &[0, 1], 2, 16.)?;
            assert_eq!(r["strict_lower"], a);
            assert_eq!(r["strict_upper"], b);
            assert_eq!(r["box_feasible"], feasible);
        }
        let r = interval(&raw(&[0., 1., 0., 3.]), &[0, 0], 2, 16.)?;
        assert!(r["strict_upper"].is_null() && r["box_feasible"] == true);
        Ok(())
    }
}
