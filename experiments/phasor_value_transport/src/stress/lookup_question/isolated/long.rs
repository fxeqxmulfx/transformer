//! Compare two write orders at increasing budgets with the learned path read alone.
use super::*;

pub fn run(seed: u32, out: &Path, hops: usize) -> Result<()> {
    let (row, task) = data()?;
    let cfg = task_config(&row, &task, true)?;
    let (weights, point) = checkpoint(seed, out, &row, &cfg)?;
    let geometry = Geometry::new(&cfg, &weights);
    let path = cfg
        .head
        .peer_read()
        .and_then(|h| h.path)
        .ok_or("no local path branch")?;
    let original = &weights[coefficients::table_range(&cfg)?];
    let floor = 1. / original.len() as f64;
    let maximum = original.iter().copied().map(f32::abs).fold(0., f32::max);
    let denominator = f64::from(maximum).powi(2) + floor;
    let table: Vec<_> = original
        .iter()
        .map(|&w| crate::typed_lookup::scaled_weight(w, floor, denominator, path.gain_power.into()))
        .collect();
    let mut conditions = BTreeMap::new();
    for mode in [
        data::long::Relations::Shared,
        data::long::Relations::Rebound,
    ] {
        let rows = data::long::long(hops, mode)?;
        let mut walk = BTreeMap::new();
        for ordered in [false, true] {
            let mut reports = BTreeMap::new();
            let mut reference = Vec::new();
            for passes in [hops, 2 * hops, 4 * hops] {
                let mut ties = 0;
                let predictions: Vec<_> = rows
                    .iter()
                    .map(|e| {
                        let (token, tied) = geometry.answer(&e.prompt, passes, &table, ordered);
                        ties += usize::from(tied);
                        token as i64
                    })
                    .collect();
                if passes == hops {
                    reference.clone_from(&predictions);
                }
                let mut counts = [0; 4];
                let mut same = 0;
                for ((&a, &b), e) in reference.iter().zip(&predictions).zip(&rows) {
                    let target = e.answer[0];
                    counts[2 * usize::from(a == target) + usize::from(b == target)] += 1;
                    same += usize::from(a == b);
                }
                let [neither, newly, lost, retained] = counts;
                reports.insert(passes, json!({"correct":newly+retained,"total":rows.len(),"ties":ties,
                    "retained":retained,"new":newly,"lost":lost,"neither":neither,"same_token":same}));
            }
            walk.insert(if ordered { "ordered" } else { "latest" }, reports);
        }
        conditions.insert(
            mode.name(),
            json!({"split_sha256":data::split_digest(&rows)?,"walk":walk}),
        );
    }
    println!(
        "{}",
        json!({"evaluation":"isolated final-prefix read at unit gain, full tied vocabulary on CPU",
      "hops":hops,"seed":seed,"step":point["step"],"weights_sha256":weight_digest(&weights),
      "protocol_sha256":row["catalog_sha256"],"model":MODEL,"exponent":path.gain_power,"conditions":conditions})
    );
    Ok(())
}
