//! Verify frozen weights before opening the separately preserved next-update evidence.
use crate::{
    ANCHORS, COHORTS, Result, SEEDS,
    analysis::{self, Executed},
    evidence::{Ledger, hash, require},
    state,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Seek, SeekFrom},
    path::Path,
    time::Instant,
};
fn file_hash(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut b = vec![0; 1 << 20];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
fn doubles(bytes: &[u8]) -> Result<Vec<f64>> {
    require(bytes.len().is_multiple_of(8), "double array shape")?;
    let out: Vec<_> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect();
    require(out.iter().all(|x| x.is_finite()), "nonfinite array")?;
    Ok(out)
}
fn trace_row(path: &Path, step: usize) -> Result<Vec<f64>> {
    let mut f = fs::File::open(path)?;
    let mut header = [0; 24];
    f.read_exact(&mut header)?;
    require(
        header
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| u64::from_le_bytes(*b))
            .collect::<Vec<_>>()
            == [1, 10001, state::E as u64],
        "source trace shape",
    )?;
    f.seek(SeekFrom::Start((24 + 4 * state::E * step) as u64))?;
    let mut b = vec![0; 4 * state::E];
    f.read_exact(&mut b)?;
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b) as f64)
        .collect())
}
pub fn verify(out: &Path, roots: &[String]) -> Result<()> {
    let started = Instant::now();
    let mut ledger = Ledger::default();
    let frozen = ledger.json(&out.join("construction.json"))?;
    require(
        frozen["status"] == "frozen"
            && frozen["source_sha256"] == crate::source_hash()
            && frozen["next_state_reads"] == 0,
        "frozen construction source/status",
    )?;
    for (p, h) in frozen["inputs"].as_object().ok_or("construction ledger")? {
        require(
            file_hash(Path::new(p))? == h.as_str().unwrap(),
            "construction input changed",
        )?;
    }
    let slots = frozen["cases"].as_array().ok_or("frozen slots")?;
    require(slots.len() == 144, "frozen inventory")?;
    let mut metrics = vec![];
    for (i, c) in slots.iter().enumerate() {
        require(
            c["status"] == "frozen"
                && c["cohort"] == COHORTS[i / 48]
                && c["seed"] == SEEDS[(i % 48) / 6]
                && c["anchor"] == ANCHORS[i % 6],
            "frozen order",
        )?;
        let bytes = ledger.read(&out.join(c["metric"]["file"].as_str().ok_or("metric file")?))?;
        require(hash(&bytes) == c["metric"]["sha256"], "metric hash changed")?;
        let f: Value = serde_json::from_slice(&bytes)?;
        let old = ledger.read(Path::new(f["old_state"].as_str().ok_or("old state")?))?;
        require(hash(&old) == f["old_sha256"], "old state changed")?;
        let t = ANCHORS[i % 6];
        let old = state::decode(&old, t)?;
        let active = serde_json::from_value::<Vec<bool>>(f["active"].clone())?;
        let (alpha, r) = state::metric(&old, &active, t);
        let eta = state::rate(COHORTS[i / 48], t);
        require(
            f["alpha"] == alpha
                && f["r"] == json!(r)
                && f["eta"] == eta
                && f["s"] == eta * alpha / (1. - 0.9f64.powi(t as i32 + 1)),
            "metric reconstruction",
        )?;
        metrics.push(f);
    }
    // Every frozen metric is verified before the first #206 report or next array opens.
    let mut cases = vec![];
    for (ci, cohort) in COHORTS.into_iter().enumerate() {
        let root = Path::new(&roots[ci]);
        let ce = ledger.json(&root.join("report.json"))?;
        let verified = ledger.json(&root.join("verification.json"))?;
        require(
            ce["status"] == "pass"
                && ce["cohort"] == cohort
                && ce["source_sha256"]
                    == "0c657c116402b8faf2d5b65ef0c940a5fc3c571a8bd0371e0d5b721fb8000ba8"
                && verified["passed"] == true
                && verified["cases"] == 48
                && verified["source_sha256"] == ce["source_sha256"],
            "#206 audit identity",
        )?;
        for (p, h) in ce["inputs"].as_object().ok_or("#206 input ledger")? {
            require(
                file_hash(Path::new(p))? == h.as_str().unwrap(),
                "#206 source input changed",
            )?;
        }
        let cc = ce["cases"].as_array().ok_or("#206 cases")?;
        require(cc.len() == 48, "#206 inventory")?;
        for (i, c) in cc.iter().enumerate() {
            let seed = SEEDS[i / 6];
            let anchor = ANCHORS[i % 6];
            let mut result = json!({"cohort":cohort,"seed":seed,"anchor":anchor});
            let computed = (|| -> Result<Value> {
                require(
                    started.elapsed().as_secs_f64() + frozen["seconds"].as_f64().unwrap() < 600.,
                    "audit cap",
                )?;
                require(
                    c["status"] == "pass"
                        && c["seed"] == seed
                        && c["anchor"] == anchor
                        && c["optimizer"]["passed"] == true,
                    "#206 case identity",
                )?;
                let f = &metrics[ci * 48 + i];
                let case = root.join(format!("seed{seed}-anchor{anchor}"));
                let mut get = |name: &str| -> Result<Vec<u8>> {
                    let b = ledger.read(&case.join(name))?;
                    require(hash(&b) == c["evidence"][name], "#206 case evidence hash")?;
                    Ok(b)
                };
                let next = state::decode(&get("next.bin")?, anchor + 1)?;
                let old = state::decode(&fs::read(f["old_state"].as_str().unwrap())?, anchor)?;
                let a = Executed {
                    features: doubles(&get("features.bin")?)?,
                    partial: doubles(&get("readout.bin")?)?,
                    full: doubles(&get("full.bin")?)?,
                    rows: serde_json::from_slice(&get("batch.json")?)?,
                    norm: c["norm"].as_f64().ok_or("source norm")?,
                };
                let trace = Path::new(ce["source"].as_str().unwrap())
                    .join(format!("seed{seed}/embeddings.bin"));
                require(
                    old.p == trace_row(&trace, anchor)? && next.p == trace_row(&trace, anchor + 1)?,
                    "original trace match",
                )?;
                let v = analysis::assess(&old, &next, f, &a)?;
                require(
                    v["q"] == c["diagnosis"]["q"] && v["eta"] == c["diagnosis"]["eta"],
                    "source CE/schedule match",
                )?;
                Ok(v)
            })();
            match computed {
                Ok(v) => {
                    result["status"] = json!("pass");
                    result["diagnosis"] = v;
                }
                Err(e) => {
                    result["status"] = json!("error");
                    result["error"] = json!(e.to_string());
                }
            }
            cases.push(result);
        }
        eprintln!(
            "metric audit cohort={cohort} slots={}/144 elapsed={:.2}s",
            cases.len(),
            started.elapsed().as_secs_f64()
        );
    }
    ledger.verify()?;
    let failed = cases.iter().filter(|c| c["status"] != "pass").count();
    let total = started.elapsed().as_secs_f64() + frozen["seconds"].as_f64().unwrap();
    println!(
        "{}",
        json!({"passed":failed==0 && total<=600.,"failed":failed,"cases":cases,
        "seconds":started.elapsed().as_secs_f64(),"combined_seconds":total,
        "construction_sha256":hash(&fs::read(out.join("construction.json"))?),
        "source_sha256":crate::source_hash(),"inputs":ledger.0})
    );
    require(
        failed == 0 && total <= 600.,
        "metric audit failed; every slot retained",
    )
}
