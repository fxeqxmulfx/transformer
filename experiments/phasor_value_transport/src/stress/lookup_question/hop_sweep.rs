//! Compare free generation on the same synthetic questions at different read budgets.
use super::*;

pub(super) fn set_budget(cfg: &mut Config, passes: usize) -> Result<()> {
    let head::Head::PeerTriplePhase(read) = &mut cfg.head else {
        return Err("a read-count change requires a repeated associative read".into());
    };
    let path = read
        .path
        .as_mut()
        .ok_or("checkpoint has no repeated associative read")?;
    if passes == 0 {
        return Err("supply a positive number of shared reads".into());
    }
    path.passes = passes;
    Ok(())
}

pub(super) fn generated(engine: &mut Engine, examples: &[Example]) -> Result<Vec<Option<i64>>> {
    let mut predictions = Vec::with_capacity(examples.len());
    let metric = metrics::evaluate_with_completions(engine, examples, true, |i, tokens| {
        if i != predictions.len() {
            return Err("generated answers changed question order".into());
        }
        predictions.push(match tokens {
            [token] => Some(i64::from(*token)),
            _ => None,
        });
        Ok(())
    })?;
    let correct = predictions
        .iter()
        .zip(examples)
        .filter(|(p, e)| **p == Some(e.answer[0]))
        .count();
    if examples.len() != predictions.len() || metric["exact"] != correct {
        return Err("per-question answers disagree with the free-generation scorer".into());
    }
    Ok(predictions)
}

pub fn run(seed: u32, out: &Path, passes: Option<usize>, compare: Option<usize>) -> Result<()> {
    let (row, mut task) = data()?;
    let mut cfg = task_config(&row, &task, true)?;
    let (weights, point) = checkpoint(seed, out, &row, &cfg)?;
    let trained = cfg.head.peer_read().and_then(|r| r.path).map(|p| p.passes);
    if let Some(n) = passes {
        set_budget(&mut cfg, n)?;
    }
    let evaluated = passes.or(trained);
    let split_hashes = data::complete_hops(&mut task)?;
    let mut results = BTreeMap::new();
    let mut originals = BTreeMap::new();
    {
        let mut engine = Engine::for_scoring(cfg.clone(), &weights)?;
        for name in std::iter::once(task.selection.clone())
            .chain(split_hashes.keys().map(|h| format!("test/hops-{h}")))
        {
            let examples = &task.splits[&name];
            let predictions = generated(&mut engine, examples)?;
            let correct = predictions
                .iter()
                .zip(examples)
                .filter(|(p, e)| **p == Some(e.answer[0]))
                .count();
            results.insert(
                name.clone(),
                json!({"correct":correct,"total":examples.len()}),
            );
            originals.insert(name, predictions);
        }
    }
    let validation = &results[&task.selection];
    if evaluated == trained && validation["correct"] != point["correct"] {
        return Err("scheduled validation changed at the saved read count".into());
    }
    let scores: BTreeMap<_, _> = split_hashes
        .keys()
        .map(|h| (*h, &results[&format!("test/hops-{h}")]))
        .collect();
    let mut report = json!({"study":NAME,"model":MODEL,"seed":seed,"step":point["step"],"kernite":KERNITE_VERSION,
        "weights_sha256":weight_digest(&weights),"training_protocol_sha256":row["catalog_sha256"],
        "trained_passes":trained,"evaluated_passes":evaluated,"split_sha256":split_hashes,
        "validation":validation,"synthetic_by_hop":scores});
    if let Some(n) = compare {
        set_budget(&mut cfg, n)?;
        let mut engine = Engine::for_scoring(cfg, &weights)?;
        let mut pairs = BTreeMap::new();
        for (name, first) in originals {
            let rows = &task.splits[&name];
            let second = generated(&mut engine, rows)?;
            let mut counts = [0; 4];
            let mut same = 0;
            for ((&a, b), e) in first.iter().zip(second).zip(rows) {
                let gold = Some(e.answer[0]);
                counts[2 * usize::from(a == gold) + usize::from(b == gold)] += 1;
                same += usize::from(a == b && a.is_some());
            }
            let [neither, gained, lost, retained] = counts;
            pairs.insert(
                name,
                json!({"retained_correct":retained,"lost_correct":lost,"newly_correct":gained,
                "neither_correct":neither,"same_generated_token":same,"total":rows.len()}),
            );
        }
        report["comparison_passes"] = json!(n);
        report["paired_read_budgets"] = json!(pairs);
    }
    println!("{report}");
    Ok(())
}
