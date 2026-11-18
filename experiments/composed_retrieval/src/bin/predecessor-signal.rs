//! CM120. Query one new content comparison on the original training
//! rows of frozen CM117 models. Both content reads share the coefficient.
#[path = "shared/training.rs"]
mod training;
use baseline::data::Example;
use composed_retrieval::{Result, data, model::Model, operator::Tables, train};
use serde_json::{Value, json};

const PRODUCER: &str = "e5bb48c79b86db3e8274211aeea73558162513eb";
const GRID: [f64; 7] = [-4., -2., -1., 0., 1., 2., 4.];

fn intervention(parent: &Model, rows: &[Example], raw: &[u8], after: &Value) -> Result<Value> {
    if !parent.learned_product_radii
        || parent.predecessor_match
        || parent.signed_profile
        || parent.context <= 2
    {
        return Err("expected the frozen CM117 sequence model".into());
    }
    let mut m = parent.clone();
    m.predecessor_match = true;
    m.weights.push(0.);
    let coefficient = m
        .validate()?
        .predecessor
        .ok_or("missing predecessor coefficient")?;
    let samples = train::samples(&m, rows)?;
    let batch: Vec<_> = samples.iter().flatten().collect();
    let lengths = training::lengths(&batch, rows);
    let repeated = batch
        .iter()
        .filter(|s| {
            let memory = &s.prefix[..s.prefix.len().saturating_sub(1)];
            memory.windows(2).enumerate().any(|(j, left)| {
                memory[j + 1..]
                    .windows(2)
                    .any(|right| left[1] == right[1] && left[0] != right[0])
            })
        })
        .count();
    let t = Tables::new(&m, &lengths)?;
    let slope = train::analytic(&m, &t, &batch)[coefficient];
    let step = 1. / 65536.;
    let mut loss = |t| -> Result<f64> {
        m.weights[coefficient] = t;
        train::metrics(&m, &samples)?["ce"]
            .as_f64()
            .ok_or_else(|| "missing CE".into())
    };
    let numerical = (loss(step)? - loss(-step)?) / (2. * step);
    if !numerical.is_finite()
        || !slope.is_finite()
        || (slope - numerical).abs() > 2e-7 * (1. + slope.abs())
    {
        return Err(format!(
            "predecessor derivative differs from queried training CE: {slope} vs {numerical}"
        )
        .into());
    }
    let mut grid = Vec::new();
    for value in GRID {
        m.weights[coefficient] = value;
        let mut score =
            training::scores(&m, &lengths, rows, (value == 0.).then_some((raw, after)))?;
        score["coefficient"] = json!(value);
        grid.push(score);
    }
    Ok(
        json!({"targets":batch.len(),"prefixes_with_repeated_key_and_distinct_predecessors":repeated,
        "ce_derivative_at_zero":slope,"forward_queried_derivative":numerical,"training":grid}),
    )
}

fn main() -> Result<()> {
    training::rewrite_sweep(PRODUCER, data::RADII_PROTOCOL, intervention)
}
