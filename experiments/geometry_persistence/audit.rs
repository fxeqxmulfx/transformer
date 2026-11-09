//! Independent forward evaluation; this checker never calls the forecast constructor.
use crate::{
    Result, SEEDS, WINDOWS, algebra,
    endpoint::{self, N, P, TRACKS},
    evidence::{hash, require, scalar, word},
    model, trace,
};
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};
fn read(p: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(p)?)?)
}
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 * (1. + a.abs().max(b.abs()))
}
fn below(a: f64, b: f64) -> bool {
    a <= b + 1e-8 * (1. + a.abs().max(b.abs()))
}
fn positive(a: f64) -> bool {
    a > 1e-8 * (1. + a.abs())
}
pub fn case(out: &Path, anchor: u64, end: u64, training: &[bool]) -> Result<Value> {
    let meta = read(&out.join("frozen.json"))?;
    let b = fs::read(out.join("bounds.bin"))?;
    require(
        b.len() == 8 * (4 + N * (P + 9))
            && (0..4).map(|i| word(&b, i)).collect::<Vec<_>>() == [1, N as u64, P as u64, 4]
            && hash(&b) == meta["bounds_sha256"]
            && meta["frozen_before_endpoint_evaluation"] == true,
        "invalid frozen evidence",
    )?;
    let old_root = Path::new(meta["old_root"].as_str().ok_or("missing old root")?);
    let next_root = Path::new(meta["next_root"].as_str().ok_or("missing next root")?);
    let old = endpoint::parameters(
        &fs::read(old_root.join("state.bin"))?,
        anchor,
        &meta["old_state_sha256"],
    )?;
    let next = endpoint::parameters(
        &fs::read(next_root.join("state.bin"))?,
        end,
        &meta["next_state_sha256"],
    )?;
    let prior = fs::read(
        Path::new(meta["old_support"].as_str().ok_or("missing prior")?).join("margins.bin"),
    )?;
    require(
        hash(&prior) == meta["prior_sha256"],
        "prior evidence changed",
    )?;
    let queries = read(&next_root.join("queries.json"))?;
    let queries = queries.as_array().ok_or("missing queries")?;
    require(
        queries.len() == N && training.len() == N,
        "query inventory changed",
    )?;
    let targets: Vec<_> = (0..N).map(|q| word(&b, 4 + q) as usize).collect();
    require(targets.iter().all(|&t| t < P), "invalid target")?;
    let rows: Vec<_> = (0..N)
        .flat_map(|q| [q as u32 / 97, q as u32 % 97, 0])
        .collect();
    let x = model::forward(model::Arm::Plain, &old, &rows);
    let y = model::forward(model::Arm::Plain, &next, &rows);
    let u = 4 + N;
    let o = u + N * P;
    let l = o + 3 * N;
    let d = l + 4 * N;
    let mut count = [0usize; 2];
    let mut correct = [0usize; 2];
    let mut cert = [[0usize; 4]; 2];
    let mut retained = [[0usize; 3]; 2];
    let mut prior_cert = [[0usize; 3]; 2];
    let mut max_excess = 0f64;
    let mut min_lower = [f64::INFINITY; 4];
    let mut margin_width = [0.; 4];
    for q in 0..N {
        let t = targets[q];
        let r = &queries[q];
        let split = usize::from(!training[q]);
        require(
            r[0] == q && r[1] == t && r[2] == split && word(&prior, 4 + q) == t as u64,
            "target route or split changed",
        )?;
        let mut actual = f64::INFINITY;
        let mut old_margin = f64::INFINITY;
        let mut max_upper = f64::NEG_INFINITY;
        let mut delta = f64::NEG_INFINITY;
        for c in (0..P).filter(|&c| c != t) {
            let old_gap = x.logits[q * P + c] - x.logits[q * P + t];
            let new_gap = y.logits[q * P + c] - y.logits[q * P + t];
            let upper = scalar(&b, u + q * P + c);
            require(
                upper.is_finite() && below(new_gap, upper),
                "future class-pair bound failed",
            )?;
            max_excess = max_excess.max((new_gap - upper) / (1. + new_gap.abs().max(upper.abs())));
            actual = actual.min(-new_gap);
            old_margin = old_margin.min(-old_gap);
            max_upper = max_upper.max(upper);
            delta = delta.max(upper - old_gap);
        }
        require(
            close(delta, scalar(&b, d + q)) && close(-max_upper, scalar(&b, l + q)),
            "frozen reduction changed",
        )?;
        let gpu = r[4].as_f64().ok_or("missing GPU margin")?;
        let right = r[3] == t;
        count[split] += 1;
        correct[split] += usize::from(right);
        for m in 0..4 {
            let lower = scalar(&b, l + m * N + q);
            require(
                lower.is_finite() && below(lower, actual) && below(lower, gpu),
                "future minimum margin bound failed",
            )?;
            let pass = positive(lower);
            require(!pass || right, "false future certificate")?;
            cert[split][m] += usize::from(pass);
            if split == 1 {
                min_lower[m] = min_lower[m].min(lower);
                margin_width[m] += actual - lower;
            }
            if m > 0 {
                let old_lower = scalar(&b, o + (m - 1) * N + q);
                require(
                    old_lower.to_bits() == scalar(&prior, 4 + N + (m + 4) * N + q).to_bits()
                        && below(old_lower, old_margin)
                        && close(lower, old_lower - delta),
                    "old certificate transport failed",
                )?;
                let old_pass = positive(old_lower);
                prior_cert[split][m - 1] += usize::from(old_pass);
                retained[split][m - 1] += usize::from(pass && old_pass);
            }
        }
    }
    require(count == [4704, 4705], "split counts changed")?;
    let coverage:Vec<_>=(0..2).flat_map(|s|(0..4).map(move |m|(s,m))).map(|(s,m)|json!({
        "split":if s==0 {"train"} else {"validation"},"method":TRACKS[m],"count":count[s],"correct":correct[s],
        "certified":cert[s][m],"coverage":cert[s][m] as f64/count[s] as f64,
        "prior_certified":if m>0 {Some(prior_cert[s][m-1])} else {None},
        "retained":if m>0 {Some(retained[s][m-1])} else {None}})).collect();
    Ok(
        json!({"status":"pass","pair_checks":N*96,"margin_checks":N*4,"false_certificates":0,
        "max_scaled_pair_excess":max_excess,"coverage":coverage,"minimum_validation_lower":min_lower,
        "mean_validation_margin_slack":margin_width.map(|v|v/4705.)}),
    )
}
pub fn all(out: &Path) -> Result<()> {
    let started = Instant::now();
    let report = read(&out.join("report.json"))?;
    require(
        report["status"] == "pass"
            && report["declared_cases"] == 192
            && report["declared_traces"] == 24
            && report["cases"].as_array().is_some_and(|v| v.len() == 192)
            && report["traces"].as_array().is_some_and(|v| v.len() == 24),
        "incomplete audit inventory",
    )?;
    for (p, h) in report["inputs"]["files"]
        .as_object()
        .ok_or("missing ledger")?
    {
        require(
            hash(&fs::read(p)?) == h.as_str().unwrap(),
            "input ledger changed",
        )?;
    }
    algebra::threads();
    let mut slot = 0;
    for (ci, name) in ["lr001", "lr01", "settle"].into_iter().enumerate() {
        require(
            report["cohorts"][ci]["name"] == name,
            "cohort order changed",
        )?;
        let root = Path::new(
            report["cohorts"][ci]["path"]
                .as_str()
                .ok_or("missing cohort path")?,
        );
        let source = read(&root.join("report.json"))?;
        for (si, seed) in SEEDS.into_iter().enumerate() {
            let seed_root = root.join(format!("seed{seed}"));
            let initial = read(&seed_root.join("step-000000/queries.json"))?;
            let mut active = vec![false; P];
            let mut training = vec![false; N];
            for r in initial.as_array().unwrap() {
                if r[2] == 0 {
                    active[r[1].as_u64().unwrap() as usize] = true;
                    training[r[0].as_u64().unwrap() as usize] = true;
                }
            }
            let tr = trace::run(
                &seed_root,
                &source["seeds"][si],
                &active,
                &out.join(format!("{name}-seed{seed}-trace.jsonl")),
                true,
            )?;
            require(
                tr == report["traces"][ci * 8 + si]["audit"],
                "trace summary changed",
            )?;
            for (anchor, end) in WINDOWS {
                let row = &report["cases"][slot];
                require(
                    row["cohort"] == name
                        && row["seed"] == seed
                        && row["anchor"] == anchor
                        && row["end"] == end
                        && row["status"] == "pass",
                    "window inventory changed",
                )?;
                let path = out.join(format!("{name}-seed{seed}-{anchor}-{end}"));
                let result = case(&path, anchor, end, &training)?;
                for (k, v) in result.as_object().unwrap() {
                    require(row["audit"][k] == *v, "independent window summary differs")?;
                }
                let meta = read(&path.join("frozen.json"))?;
                require(
                    meta["bounds_sha256"] == row["audit"]["bounds_sha256"],
                    "result hash differs",
                )?;
                slot += 1;
            }
            eprintln!("verified persistence cohort={name} seed={seed}");
        }
    }
    let combined = report["seconds"]
        .as_f64()
        .ok_or("missing measurement time")?
        + started.elapsed().as_secs_f64();
    require(combined <= 1800., "combined postprocessing cap exceeded")?;
    println!(
        "{}",
        json!({"passed":true,"cases":slot,"traces":24,"prefixes":96000,"pair_checks":slot*N*96,
        "margin_checks":slot*N*4,"false_certificates":0,"seconds":started.elapsed().as_secs_f64(),"combined_postprocessing_seconds":combined})
    );
    Ok(())
}
