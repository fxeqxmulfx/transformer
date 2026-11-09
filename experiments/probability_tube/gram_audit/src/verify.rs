//! Verify sealed old-only construction before reading authenticated future states.
use crate::{
    CAP, Result,
    application::Backend,
    classes::Partition,
    endpoint,
    evidence::{self, Ledger, require},
    gpu,
    infrastructure::storage,
    prediction, provenance, sources,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
struct Audit {
    dirs: [PathBuf; 2],
    roots: Vec<PathBuf>,
    source: [Value; 2],
    previous: [Value; 2],
    hashes: BTreeMap<String, String>,
}
impl Audit {
    fn load(frozen: &Value, ledger: &mut Ledger) -> Result<Self> {
        let dirs: [PathBuf; 2] = serde_json::from_value(frozen["source_forecasts"].clone())?;
        let roots: Vec<PathBuf> = serde_json::from_value(frozen["roots"].clone())?;
        let source = sources::pair(&dirs, ledger)?;
        require(
            source[0]["roots"] == frozen["roots"],
            "changed cohort roots",
        )?;
        let previous = [
            ledger.json(&dirs[0].join("verification-01.json"))?,
            ledger.json(&dirs[1].join("verification-01.json"))?,
        ];
        let mut hashes = BTreeMap::new();
        for family in 0..2 {
            let report = &previous[family];
            require(
                report["status"] == "pass"
                    && report["cases"].as_array().is_some_and(|v| v.len() == 144)
                    && report["provenance"]["source_sha256"] == sources::HASHES[family]
                    && report["construction_sha256"]
                        == evidence::hash(&fs::read(dirs[family].join("construction.json"))?),
                "source verification identity",
            )?;
            for input in report["inputs"]
                .as_array()
                .ok_or("source verification ledger")?
            {
                let path = input["path"]
                    .as_str()
                    .ok_or("source input path")?
                    .to_owned();
                let hash = input["sha256"]
                    .as_str()
                    .ok_or("source input hash")?
                    .to_owned();
                if let Some(old) = hashes.insert(path, hash.clone()) {
                    require(old == hash, "conflicting source input hashes")?;
                }
            }
        }
        for (path, hash) in &hashes {
            require(
                evidence::hash(&ledger.read(Path::new(path))?) == *hash,
                "changed source verification input",
            )?;
        }
        Ok(Self {
            dirs,
            roots,
            source,
            previous,
            hashes,
        })
    }
}
fn slot(
    audit: &Audit,
    out: &Path,
    index: usize,
    spec: &Value,
    clock: (Instant, f64),
    ledger: &mut Ledger,
) -> Result<Value> {
    let (started, charged) = clock;
    let seed = spec["seed"].as_u64().ok_or("seed")?;
    let anchor = spec["anchor"].as_u64().ok_or("anchor")? as usize;
    let f = sources::payload(out, spec, ledger)?;
    let (old, data, plan) = sources::old(&audit.roots[index / 48], seed, anchor, ledger)?;
    let source = [
        sources::payload(&audit.dirs[0], &audit.source[0]["cases"][index], ledger)?,
        sources::payload(&audit.dirs[1], &audit.source[1]["cases"][index], ledger)?,
    ];
    require(
        f == prediction::make(
            &old,
            &data,
            &plan,
            [&source[0], &source[1]],
            started,
            charged,
        )?,
        "old-only reconstruction",
    )?;
    // The payload is sealed and reconstructed before any future state for this case.
    let mut e = gpu::engine(&old, &data, &plan)?;
    let old_gpu = gpu::snapshot(&e, &data)?;
    let previous = [
        &audit.previous[0]["cases"][index],
        &audit.previous[1]["cases"][index],
    ];
    for p in previous {
        for key in ["cohort", "seed", "anchor"] {
            require(p[key] == spec[key], "source result case")?;
        }
        require(p["status"] == "pass", "source case verification")?;
    }
    require(
        previous[0]["next_step"] == previous[1]["next_step"],
        "changed signed next-step decomposition",
    )?;
    let ctx = endpoint::Case {
        data: &data,
        plan: &plan,
        source: [&source[0], &source[1]],
        forecast: &f,
        previous,
        old_gpu: &old_gpu,
        partition: Partition::new(sources::support(&data), 128),
    };
    let mut endpoints = vec![];
    for h in spec["horizons"].as_array().ok_or("horizons")? {
        let horizon = h.as_u64().ok_or("horizon")? as usize;
        let result = (|| -> Result<Value> {
            require(
                charged + started.elapsed().as_secs_f64() < CAP,
                "verification budget",
            )?;
            let path = audit.roots[index / 48]
                .join(format!(
                    "seed{seed}/checkpoints/step-{:06}.bin",
                    anchor + horizon
                ))
                .canonicalize()?;
            let bytes = ledger.read(&path)?;
            require(
                audit
                    .hashes
                    .get(path.to_str().ok_or("checkpoint path")?)
                    .is_some_and(|h| *h == evidence::hash(&bytes)),
                "authenticated future checkpoint",
            )?;
            let (step, next) = storage::decode(&bytes, crate::domain::model::Arm::Plain.count())?;
            require(step == anchor + horizon, "future time")?;
            e.restore(&next)?;
            endpoint::check(&ctx, &next, &e, horizon)
        })();
        endpoints.push(match result {
            Ok(v) => v,
            Err(e) => json!({"horizon":horizon,"status":"error","error":e.to_string()}),
        });
    }
    let pass = endpoints.iter().all(|e| e["status"] == "pass");
    Ok(
        json!({"cohort":spec["cohort"],"seed":seed,"anchor":anchor,"status":if pass{"pass"}else{"fail"},
        "endpoints":endpoints,"next_step":previous[0]["next_step"]}),
    )
}
pub fn run(out: &Path) -> Result<()> {
    let started = Instant::now();
    let mut ledger = Ledger::default();
    let frozen = sources::sealed(out, &mut ledger)?;
    require(
        frozen["source_sha256"] == provenance::source_hash(),
        "constructor/verifier source mismatch",
    )?;
    let mut charged = sources::number(&frozen["seconds"])?;
    let mut attempt = 1;
    while out.join(format!("verification-{attempt:02}.json")).exists() {
        let old = ledger.json(&out.join(format!("verification-{attempt:02}.json")))?;
        require(old["status"] != "running", "verification already running")?;
        charged += sources::number(&old["seconds"])?;
        attempt += 1;
    }
    let path = out.join(format!("verification-{attempt:02}.json"));
    let mut report = json!({"status":"running","construction_sha256":evidence::hash(&fs::read(out.join("construction.json"))?),
        "charged_seconds":charged,"cap_seconds":CAP,"cases":frozen["cases"]});
    evidence::write(&path, &report)?;
    let result = (|| -> Result<()> {
        require(charged < CAP, "audit budget exhausted")?;
        report["provenance"] = provenance::metadata()?;
        let audit = Audit::load(&frozen, &mut ledger)?;
        for index in 0..144 {
            let spec = &frozen["cases"][index];
            report["cases"][index] =
                match slot(&audit, out, index, spec, (started, charged), &mut ledger) {
                    Ok(v) => v,
                    Err(e) => {
                        json!({"cohort":spec["cohort"],"seed":spec["seed"],"anchor":spec["anchor"],
                    "status":"error","error":e.to_string(),"horizons":spec["horizons"]})
                    }
                };
            evidence::write(&path, &report)?;
            eprintln!(
                "verified slot={index} status={} cumulative={:.1}s",
                report["cases"][index]["status"],
                charged + started.elapsed().as_secs_f64()
            );
        }
        ledger.verify()?;
        Ok(())
    })();
    if let Err(e) = result {
        report["error"] = json!(e.to_string());
    }
    let failed = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["status"] != "pass")
        .count();
    let seconds = started.elapsed().as_secs_f64();
    let pass = failed == 0 && report["error"].is_null() && charged + seconds <= CAP;
    report["status"] = json!(if pass { "pass" } else { "fail" });
    report["failed"] = json!(failed);
    report["seconds"] = json!(seconds);
    report["total_seconds"] = json!(charged + seconds);
    report["inputs"] = json!(ledger.0);
    evidence::write(&path, &report)?;
    require(pass, "verification failed; complete inventory retained")
}
