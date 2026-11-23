//! Compare soft retrieval with maximum-score retrieval on the identical prefix.
use super::*;
pub fn run(seed: u32, out: &Path) -> Result<()> {
    let (row, mut task) = data()?;
    let mut cfg = task_config(&row, &task, true)?;
    if !cfg
        .head
        .peer_read()
        .and_then(|r| r.path)
        .is_some_and(|p| p.soft_address)
    {
        return Err("the index comparison requires a scalar address checkpoint".into());
    }
    let (weights, point) = checkpoint(seed, out, &row, &cfg)?;
    let digest = weight_digest(&weights);
    let hashes = data::complete_hops(&mut task)?;
    cfg.max_batch = 1;
    let mut engine = Engine::for_scoring(cfg, &weights)?;
    let mut comparisons = BTreeMap::new();
    for split in [
        "validation",
        "test/hops-1",
        "test/hops-2",
        "test/hops-5",
        "test/hops-10",
    ] {
        let rows = &task.splits[split];
        crate::soft_address::index_audit::begin(false);
        crate::soft_address::audit::start();
        let soft = hop_sweep::generated(&mut engine, rows);
        let quality = crate::soft_address::audit::finish();
        let timing = crate::soft_address::index_audit::finish();
        let soft = soft?;
        crate::soft_address::index_audit::begin(true);
        let hard = hop_sweep::generated(&mut engine, rows);
        crate::soft_address::index_audit::finish();
        let hard = hard?;
        let mut cells = [0; 4];
        let mut same = 0;
        for ((a, b), sample) in soft.iter().zip(&hard).zip(rows) {
            let truth = Some(sample.answer[0]);
            cells[2 * usize::from(*a == truth) + usize::from(*b == truth)] += 1;
            same += usize::from(a == b);
        }
        let [neither, gained, lost, both] = cells;
        comparisons.insert(split,json!({"soft_correct":both+lost,"indexed_correct":both+gained,
          "retained_correct":both,"lost_correct":lost,"newly_correct":gained,"neither_correct":neither,
          "same_answer":same,"total":rows.len(),"latency":timing,"soft_payload":quality}));
    }
    if comparisons["validation"]["soft_correct"] != point["correct"] {
        return Err("soft checkpoint replay changed".into());
    }
    println!(
        "{}",
        json!({"study":NAME,"model":MODEL,"seed":seed,"step":point["step"],"kernite":KERNITE_VERSION,
        "weights_sha256":digest,"protocol_sha256":row["catalog_sha256"],"split_sha256":hashes,"by_split":comparisons})
    );
    Ok(())
}
