//! Compare sealed outcomes with the CM115 reference. Remove output offsets
//! only after fitting and count the counterfactual on original training rows.
#[path = "shared/audit.rs"]
mod archived;
use baseline::{
    channel::{argmax, cross_entropy},
    data::Example,
    hash,
};
use bilinear::guard;
use composed_retrieval::{Result, data, model::Model, operator::Tables, train};
use learned_retrieval::cohort::read;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

fn ablation(mut m: Model, rows: &[Example]) -> Result<Value> {
    let l = m.validate()?;
    let b = &mut m.weights[l.output_offsets];
    let bound = m.output_offset_bound.ok_or("model has no output offsets")?;
    let max = argmax(b);
    let min = b.iter().copied().fold(f64::INFINITY, f64::min);
    let offsets = json!({"min":min,"max":b[max],"max_class":max,
        "at_bound":b.iter().filter(|v| v.abs()==bound).count(),"class_zero":b[0]});
    b.fill(0.);
    let samples = train::samples(&m, rows)?;
    let lengths: BTreeSet<_> = samples
        .iter()
        .flatten()
        .map(|s| s.prefix.len().saturating_sub(1))
        .collect();
    let t = Tables::new(&m, &lengths.into_iter().collect::<Vec<_>>())?;
    let mut correct = 0;
    let mut count = 0;
    let mut loss = 0.;
    let mut classes = BTreeMap::<_, [usize; 2]>::new();
    for s in samples.iter().flatten() {
        let z = t.logits(&m, &t.pass(&m, &s.prefix).hidden);
        let hit = usize::from(argmax(&z) == s.target);
        let v = classes.entry(s.target).or_default();
        v[0] += hit;
        v[1] += 1;
        correct += hit;
        count += 1;
        loss += cross_entropy(&z, s.target);
    }
    Ok(
        json!({"learned_offsets":offsets,"with_offsets_zero":{"correct":correct,"targets":count,
        "ce":loss/count as f64,"classes":classes}}),
    )
}

fn splits(report: &Value, key: &str) -> Result<BTreeMap<String, Value>> {
    let mut out = BTreeMap::new();
    for (name, r) in report["cells"][key]["splits"]
        .as_object()
        .ok_or("missing raw scores")?
    {
        let m = &r["metrics"];
        out.insert(
            name.clone(),
            json!({"raw":[m["correct"],m["targets"]],"exact":[m["exact"],m["examples"]]}),
        );
    }
    Ok(out)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [work, storage, four, family, changed] = args.as_slice() else {
        return Err(
            "expected WORKSPACE STORAGE LOGODDS_FOUR LOGODDS_ARITHMETIC OFFSET_OUTPUT".into(),
        );
    };
    let (work, storage) = (Path::new(work), Path::new(storage));
    guard::admit(1 << 30, work, storage)?;
    let roots = [four, family, changed].map(Path::new);
    let defs = [
        data::TRAINING_PROTOCOL,
        data::JOINT_PROTOCOL,
        data::PROTOCOL,
    ];
    let mut audits = Vec::new();
    for ((root, definition), n) in roots.into_iter().zip(defs).zip([4, 48, 51]) {
        for arm in ["original", "repeat"] {
            let manifest = read(&root.join(arm).join("run/sealed/manifest.json"))?;
            if manifest["identity"]["protocol_sha256"] != hash(definition.as_bytes()) {
                return Err("unexpected reference or output-offset study".into());
            }
        }
        let a = archived::audit(work, storage, root)?;
        if a["summary"]["measured"] != n {
            return Err("unexpected cohort size".into());
        }
        audits.push(a);
    }
    let p = data::protocol()?;
    let mut cases = BTreeMap::new();
    let mut initial_files = 0;
    for (id, spec) in &p.cases {
        let baseline = usize::from(spec.source_fingerprints.is_some());
        let (task, pins) = data::original(&p, id)?;
        let [a, b] = [roots[baseline], roots[2]].map(|root| root.join("original/run").join(id));
        let (ra, rb) = (
            read(&a.join("scores/report.json"))?,
            read(&b.join("scores/report.json"))?,
        );
        let mut cells = BTreeMap::new();
        for mode in &p.modes {
            for seed in &p.seeds {
                let cell = format!("{mode}-{seed}");
                let initial = format!("{cell}-initial");
                let end = format!("{cell}-final");
                let receipts = [&a, &b].map(|dir| read(&dir.join(&cell).join("receipt.json")));
                let [old, new] = receipts;
                let (old, new) = (old?, new?);
                let (u, v) = (&old["training"], &new["training"]);
                if old["data_fingerprints"] != pins
                    || new["data_fingerprints"] != pins
                    || [
                        "updates",
                        "examples",
                        "targets",
                        "batch_sha256",
                        "min_visits",
                        "max_visits",
                    ]
                    .iter()
                    .any(|&field| u[field].is_null() || u[field] != v[field])
                {
                    return Err(format!("training inputs or visits changed for {id}/{cell}").into());
                }
                let extra = if mode == "forward" {
                    2 * spec.vocabulary * p.updates
                } else {
                    0
                };
                if v["forward_loss_queries"].as_u64().ok_or("new queries")?
                    != u["forward_loss_queries"]
                        .as_u64()
                        .ok_or("reference queries")?
                        + extra as u64
                {
                    return Err("unexpected number of additional forward CE queries".into());
                }
                for split in task.splits.keys() {
                    let sha = &ra["cells"][&initial]["splits"][split]["raw_sha256"];
                    if sha.is_null() || *sha != rb["cells"][&initial]["splits"][split]["raw_sha256"]
                    {
                        return Err(
                            format!("initial raw logits changed for {id}/{cell}/{split}").into(),
                        );
                    }
                    initial_files += 1;
                }
                let model = serde_json::from_slice(&fs::read(b.join(&cell).join("model.json"))?)?;
                let effect = ablation(model, &task.splits["train"])?;
                if effect["with_offsets_zero"]["targets"] != v["after"]["targets"] {
                    return Err("counterfactual training denominator changed".into());
                }
                cells.insert(
                    cell,
                    json!({"reference":splits(&ra,&end)?,"current":splits(&rb,&end)?,
                    "fitted_train":v["after"],"ablation":effect}),
                );
            }
        }
        cases.insert(
            id,
            json!({"reference":audits[baseline]["cases"][id]["current"],
            "current":audits[2]["cases"][id]["current"],"cells":cells}),
        );
    }
    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &json!({"summary":audits[2]["summary"],
        "identical_initial_score_files":initial_files,"cases":cases}),
    )?;
    println!();
    Ok(())
}
