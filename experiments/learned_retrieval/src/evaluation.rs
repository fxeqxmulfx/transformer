//! Roles enter here only after sealing every endpoint. Every score uses the
//! original full-class predictor. Pair-restricted summaries are diagnostic.
use crate::{
    Result,
    binding_counts::{Context, Counts, Observation},
    cohort::{self, Protocol},
    model::{Model, dot},
};
use baseline::{
    channel::argmax,
    data::Example,
    metrics::{self, Predictor},
};
use bilinear::{
    experiment::{hash, records, write_json},
    guard,
    train::Record,
};
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

struct Raw<'a> {
    model: &'a Model,
    bytes: RefCell<Vec<u8>>,
}
impl Predictor for Raw<'_> {
    fn vocab(&self) -> usize {
        self.model.vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let z = self
            .model
            .logits(prefix)
            .expect("original prefix was shape checked");
        let mut raw = self.bytes.borrow_mut();
        for x in &z {
            raw.extend_from_slice(&x.to_le_bytes());
        }
        z
    }
}
type Pairs = BTreeSet<(i64, usize)>;
fn seen(records: &[Record<'_>]) -> Result<(Pairs, Pairs)> {
    let mut queried = Pairs::new();
    let mut stored = Pairs::new();
    for record in records {
        let c = Context::new(record)?;
        queried.insert((*record.prefix.last().ok_or("query")?, record.target));
        for (key, value) in [record.prefix[1], record.prefix[3]]
            .into_iter()
            .zip(c.values)
        {
            stored.insert((key, value));
        }
    }
    Ok((queried, stored))
}

#[derive(Default)]
struct Routes {
    a: f64,
    b: f64,
    ab: f64,
    positive: usize,
    considered: usize,
}
impl Routes {
    fn observe(&mut self, m: &Model, record: &Record<'_>, ctx: &Context, z: &[f64]) -> Result<()> {
        let ids = m.objects(record.prefix)?;
        let n = ids.len() - 1;
        let j = 1 + 2 * ctx.selected;
        let k = j + 1;
        let links = m.links(n);
        let pass = m.pass(&ids, &links);
        if pass.logits != z {
            return Err("diagnostic and original complete logits differ".into());
        }
        let key_score = |pos| m.scale() * dot(m.key(ids[n]), m.key(ids[pos]));
        let link_score = |pos| m.weights[m.relative_index(j, pos)];
        let gap = |good: usize, f: &dyn Fn(usize) -> f64| {
            f(good)
                - (0..n)
                    .filter(|&i| i != good)
                    .map(f)
                    .fold(f64::NEG_INFINITY, f64::max)
        };
        let g = gap(j, &key_score);
        let h = gap(k, &link_score);
        let v = m.value(ids[k]);
        let rivals = (0..m.vocabulary).filter(|&c| c != ids[k]);
        let margin = dot(v, v)
            - rivals
                .clone()
                .map(|c| dot(m.value(c), v))
                .fold(f64::NEG_INFINITY, f64::max);
        if margin >= 0. {
            let bound =
                m.capacity * (margin - (margin + 2.) * n as f64 * ((-g).exp() + (-h).exp()));
            let actual = z[ids[k]] - rivals.map(|c| z[c]).fold(f64::NEG_INFINITY, f64::max);
            if bound > actual + 1e-10 {
                return Err("CM111 bound exceeds executed margin".into());
            }
            self.considered += 1;
            self.positive += usize::from(bound > 0.);
        }
        self.a += pass.attention[j];
        self.b += links[j * n + k];
        self.ab += pass.attention[j] * links[j * n + k];
        Ok(())
    }
}
fn split(
    model: &Model,
    examples: &[Example],
    queried: &Pairs,
    stored: &Pairs,
) -> Result<(Value, Vec<u8>)> {
    let rows = records(examples, &(0..examples.len()).collect::<Vec<_>>())?;
    if rows.len() != examples.len() {
        return Err("binding probe expects one label per example".into());
    }
    for r in &rows {
        model.objects(r.prefix)?;
    }
    let predictor = Raw {
        model,
        bytes: RefCell::new(vec![]),
    };
    let metrics = metrics::evaluate(&predictor, examples)?;
    let raw = predictor.bytes.into_inner();
    if raw.len() != rows.len() * model.vocabulary * 8 {
        return Err("one complete original output per row required".into());
    }
    let mut counts = Counts::default();
    let mut routes = Routes::default();
    let mut unseen = [[0usize; 2]; 2];
    for (record, line) in rows.iter().zip(raw.chunks_exact(model.vocabulary * 8)) {
        let logits: Vec<_> = line
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect();
        let ctx = Context::new(record)?;
        counts.observe(&Observation::new(&ctx, &logits)?);
        routes.observe(model, record, &ctx, &logits)?;
        let key = (*record.prefix.last().ok_or("query")?, record.target);
        for (set, bin) in [queried, stored].iter().zip(&mut unseen) {
            if !set.contains(&key) {
                bin[0] += 1;
                bin[1] += usize::from(argmax(&logits) == record.target);
            }
        }
    }
    if metrics["correct"] != counts.categories[0]
        || metrics["exact"] != counts.categories[0]
        || metrics["targets"] != counts.total
        || metrics["examples"] != counts.total
        || (metrics["loss"].as_f64().ok_or("CE")? - counts.full_ce / counts.total as f64).abs()
            > 2e-12
    {
        return Err("binding diagnostic disagrees with original metrics".into());
    }
    let n = rows.len() as f64;
    Ok((
        json!({"metrics":metrics,"raw99":metrics["token_accuracy"].as_f64().ok_or("raw")?>=0.99,
        "exact99":metrics["sequence_accuracy"].as_f64().ok_or("exact")?>=0.99,"raw_logits_sha256":hash(&raw),
        "binding":counts.report(),"unseen_supervised_pair":{"examples":unseen[0][0],"correct":unseen[0][1]},
        "unseen_stored_pair":{"examples":unseen[1][0],"correct":unseen[1][1]},
        "mean_key_mass":routes.a/n,"mean_link_mass":routes.b/n,"mean_path_mass":routes.ab/n,
        "nonnegative_decoder_gap":routes.considered,"positive_cm111_bound":routes.positive}),
        raw,
    ))
}
pub fn run(workspace: &Path, storage: &Path, root: &Path, replay: bool) -> Result<()> {
    guard::admit(1 << 30, workspace, storage)?;
    let p: Protocol = cohort::protocol()?;
    let cells = cohort::load(root, &p)?;
    let (row, task) = p.data()?;
    let train = &task.splits["train"];
    let (queried, stored) = seen(&records(train, &(0..train.len()).collect::<Vec<_>>())?)?;
    let output = root.join("scores");
    let dest = if replay {
        root.join("replay")
    } else {
        output.clone()
    };
    cohort::publish(&dest, |dir| {
        let mut reports = BTreeMap::new();
        for cell in cells {
            if cell.model.vocabulary != task.vocab {
                return Err("original class count changed".into());
            }
            let initial = p.initial(task.vocab, cell.seed)?;
            for (name, model) in [
                (format!("{}-initial", cell.name), &initial),
                (cell.name, &cell.model),
            ] {
                let mut splits = BTreeMap::new();
                let mut all = true;
                for (split_name, examples) in &task.splits {
                    guard::admit(1 << 27, workspace, storage)?;
                    let (report, raw) = split(model, examples, &queried, &stored)?;
                    if replay {
                        if fs::read(cohort::raw_file(&output, &name, split_name))? != raw {
                            return Err(
                                format!("{name}/{split_name} raw-logit replay mismatch").into()
                            );
                        }
                    } else {
                        fs::write(cohort::raw_file(dir, &name, split_name), &raw)?;
                    }
                    if !split_name.starts_with("train") {
                        all &= report["raw99"] == true && report["exact99"] == true;
                    }
                    splits.insert(split_name, report);
                }
                let metric = row["case"]["metric"].as_str().ok_or("original metric")?;
                let threshold = row["case"]["target"].as_f64().ok_or("original threshold")?;
                let success = splits[&task.selection]["metrics"][metric]
                    .as_f64()
                    .ok_or("primary score")?
                    >= threshold;
                reports.insert(name,json!({"original_success":success,"original_metric":metric,
                    "original_threshold":threshold,"selection":task.selection,"all_heldout_raw_and_exact99":all,"splits":splits}));
            }
        }
        let file = "scores.json";
        write_json(&dir.join(file), &reports)?;
        if replay && fs::read(dir.join(file))? != fs::read(output.join(file))? {
            return Err("complete score replay mismatch".into());
        }
        Ok(())
    })
}
