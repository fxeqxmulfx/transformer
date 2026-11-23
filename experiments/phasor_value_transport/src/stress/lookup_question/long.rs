//! Beyond-ten-hop transfer with every starting key scored on identical tables.
use super::*;
pub fn run(
    seed: u32,
    out: &Path,
    hops: usize,
    passes: usize,
    relation: &str,
    selection: Option<&str>,
    ordered: bool,
) -> Result<()> {
    let mode = data::long::Relations::parse(relation)?;
    let (row, task) = data()?;
    let mut cfg = task_config(&row, &task, true)?;
    let (mut weights, point) = checkpoint(seed, out, &row, &cfg)?;
    let digest = weight_digest(&weights);
    let intervention = if let Some(mask) = selection {
        let range = coefficients::table_range(&cfg)?;
        Some(coefficients::transform(
            &mut weights[range],
            mask,
            "linear",
        )?)
    } else {
        None
    };
    let rows = data::long::long(hops, mode)?;
    let question_hash = data::split_digest(&rows)?;
    hop_sweep::set_budget(&mut cfg, passes)?;
    let head::Head::PeerTriplePhase(routing) = &mut cfg.head else {
        return Err("a path store is required".into());
    };
    let path = routing.path.as_mut().ok_or("missing path store")?;
    if ordered && path.soft_address {
        return Err("chronological relation reads require a discrete write history".into());
    }
    path.ordered |= ordered;
    let ordered = path.ordered;
    cfg.max_length = rows
        .iter()
        .map(|e| e.prompt.len())
        .max()
        .ok_or("empty long lookup test")?;
    cfg.max_batch = task_config(&row, &task, false)?.max_batch;
    let length = cfg.max_length;
    let effective = weight_digest(&weights);
    let mut engine = Engine::for_scoring(cfg, &weights)?;
    let predictions = hop_sweep::generated(&mut engine, &rows)?;
    let mut histogram = [0; 6];
    let mut constant = 0;
    for (set, responses) in rows
        .as_chunks::<5>()
        .0
        .iter()
        .zip(predictions.as_chunks::<5>().0)
    {
        let correct = set
            .iter()
            .zip(responses)
            .filter(|(e, p)| **p == Some(e.answer[0]))
            .count();
        histogram[correct] += 1;
        constant += usize::from(responses.iter().all(|x| *x == responses[0]));
    }
    let exact = histogram
        .iter()
        .enumerate()
        .map(|(n, &count)| n * count)
        .sum::<usize>();
    let mut report = json!({"study":"synthetic-long-lookup","model":MODEL,"seed":seed,"step":point["step"],"kernite":KERNITE_VERSION,
        "training_protocol_sha256":row["catalog_sha256"],"weights_sha256":digest,"effective_weights_sha256":effective,
        "coefficient_intervention":intervention,"relations":mode.name(),"hops":hops,"evaluated_passes":passes,
        "prompt_length":length,"split_sha256":question_hash,"correct":exact,"total":rows.len(),"table_sets":rows.len()/5,
        "sets_by_correct_answers":histogram,"constant_response_sets":constant});
    if ordered {
        report["ordered_writes"] = json!(true);
    }
    println!("{report}");
    Ok(())
}
