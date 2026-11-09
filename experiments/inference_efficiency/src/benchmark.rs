//! Complete operand-request rounds, with setup charged separately and no selection.
use crate::{
    domain::Result,
    inputs::require,
    serving::{METHODS, Service},
    timer,
};
use serde_json::{Value, json};
use std::{hint::black_box, path::Path, time::Instant};
pub fn run(
    service: &mut Service<'_, '_>,
    pairs: &[[u32; 2]],
    reference: &[usize],
    cohort: &Instant,
    progress: &Path,
) -> Result<Value> {
    require(
        pairs.len() == 512 && reference.len() == 512,
        "timing set must be fixed 512 operands",
    )?;
    let mut batches = vec![];
    for batch in [1, 32, 512] {
        let mut rounds = vec![];
        let mut capped = false;
        for round in 0..12 {
            if cohort.elapsed().as_secs_f64() >= 900. {
                capped = true;
                break;
            }
            let mut times = [0.; 4];
            let mut checks = [false; 4];
            let mut errors = vec![];
            for offset in 0..4 {
                let method = (round + offset) % 4;
                let start = Instant::now();
                let result = (|| -> Result<Vec<usize>> {
                    let mut result = Vec::with_capacity(512);
                    for inputs in pairs.chunks(batch) {
                        result.extend(service.request(inputs, method)?);
                    }
                    black_box(&result);
                    Ok(result)
                })();
                times[method] = start.elapsed().as_secs_f64();
                match result {
                    Ok(v) => checks[method] = v == reference,
                    Err(e) => errors.push(json!({"method":METHODS[method],"error":e.to_string()})),
                };
                crate::infrastructure::storage::json_write(
                    progress,
                    &json!({"completed_batches":batches,"completed_rounds":rounds,
                    "batch":batch,"current_round":round,"completed_methods":offset+1,"seconds":times,"agreement":checks,"errors":errors}),
                )?;
            }
            rounds.push(json!({"round":round,"warmup":round<3,"order":(0..4).map(|o|METHODS[(round+o)%4]).collect::<Vec<_>>(),
                "seconds":times,"agreement":checks,"errors":errors}));
        }
        let measured: Vec<_> = rounds.iter().filter(|r| r["warmup"] == false).collect();
        let metrics:Vec<_>=(0..4).map(|m| {
            let times:Vec<_>=measured.iter().map(|r|r["seconds"][m].as_f64().unwrap()).collect();
            let ratios=|control:usize|timer::stats(measured.iter().map(|r|
                r["seconds"][control].as_f64().unwrap()/r["seconds"][m].as_f64().unwrap()));
            json!({"method":METHODS[m],"round_seconds":timer::stats(times.iter().copied()),
                "queries_per_second":timer::stats(times.iter().map(|s|512./s)),"paired_source_ratio":ratios(0),"paired_cached_ratio":ratios(1)})
        }).collect();
        let agreement = rounds
            .iter()
            .all(|r| r["agreement"].as_array().unwrap().iter().all(|v| v == true));
        batches.push(json!({"batch":batch,"capped":capped,"agreement":agreement,"rounds":rounds,"metrics":metrics}));
        if capped {
            break;
        }
    }
    let completed = batches.len() == 3 && batches.iter().all(|b| b["capped"] == false);
    let passed = completed && batches.iter().all(|b| b["agreement"] == true);
    Ok(json!({"passed":passed,"completed":completed,"batches":batches}))
}
pub fn setup_costs(
    benchmark: &Value,
    model: f64,
    weights: f64,
    old: f64,
    bounds: &[crate::bounds::Bound; 2],
) -> Value {
    let setup = [
        0.,
        weights,
        weights + old + bounds[0].seconds,
        weights + old + bounds[1].seconds,
    ];
    json!({"model_seconds":model,"additional_setup_seconds":setup,"batches":benchmark["batches"].as_array().unwrap().iter().map(|b|{
        let med:Vec<_>=b["metrics"].as_array().unwrap().iter().map(|m|m["round_seconds"]["median"].as_f64()).collect();
        json!({"batch":b["batch"],"methods":(0..4).map(|m|{
            let breakeven=|control:usize|match(med[control],med[m]){
                (Some(a),Some(v))if a>v=>Some(((setup[m]-setup[control]).max(0.)/(a-v)).ceil()),_=>None};
            json!({"method":METHODS[m],"cold_seconds":med[m].map(|v|model+setup[m]+v),
                "break_even_source_rounds":breakeven(0),"break_even_cached_rounds":breakeven(1)})
        }).collect::<Vec<_>>()})
    }).collect::<Vec<_>>()})
}
