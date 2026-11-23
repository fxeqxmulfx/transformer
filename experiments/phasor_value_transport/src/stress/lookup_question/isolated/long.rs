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
    let original = &weights[coefficients::trained_range(&cfg)?];
    let table = crate::typed_lookup::inference_table(original, path);
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
                        let (token, tied) = geometry.answers(
                            &e.prompt,
                            &[passes],
                            &table,
                            ordered,
                            path.agreement.then_some(path.gain_power),
                            crate::typed_lookup::Walk::Sequential,
                        )[0];
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

/// Time only the CPU path branch on actual printed prompts, with a fixed
/// synthetic hidden context and the checkpoint's normalized input embeddings.
pub fn time(
    out: &Path,
    hops: usize,
    passes: usize,
    batch: usize,
    gather: Option<&str>,
) -> Result<()> {
    if batch == 0 || passes == 0 {
        return Err("expected positive batch and pass counts".into());
    }
    if !matches!(
        gather,
        None | Some("dense-ends" | "requested-ends" | "training")
    ) {
        return Err("choose dense-ends, requested-ends or training".into());
    }
    let training = gather == Some("training");
    let (case, task, rows) = if training {
        use crate::stress::disjoint_lookup::{self, NAME, Training};
        let (case, task) = disjoint_lookup::dataset(Training::named(NAME).unwrap())?;
        (case, task, disjoint_lookup::probes(hops, NAME)?)
    } else {
        let (case, task) = data()?;
        (
            case,
            task,
            data::long::long(hops, data::long::Relations::Shared)?,
        )
    };
    let cfg = task_config(&case, &task, true)?;
    let (w, point) = checkpoint(0, out, &case, &cfg)?;
    let geometry = Geometry::new(&cfg, &w);
    let layout = crate::slot_memory::Layout::new(&cfg);
    let store = &w[architecture::memory_start(&cfg)..];
    let mut router = crate::slot_memory::Routing::new(cfg.head);
    router.geometry(&layout, store);
    let mut read = cfg
        .head
        .peer_read()
        .and_then(|h| h.path)
        .ok_or("no path branch")?;
    read.passes = passes;
    if read.soft_address {
        return Err("the read timing benchmark requires a discrete relation checkpoint".into());
    }
    let path = &store[layout.path_start.ok_or("no path weights")?..layout.end()];
    let mut output = Vec::new();
    let mut elapsed = std::time::Duration::ZERO;
    let mut adjoint = std::time::Duration::ZERO;
    for block in rows[..20].chunks(batch) {
        let length = block[0].prompt.len();
        let x: Vec<_> = block
            .iter()
            .flat_map(|e| {
                e.prompt
                    .iter()
                    .flat_map(|&t| geometry.values[t as usize].iter().copied())
            })
            .collect();
        let context: Vec<_> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| v * (1. + (i % 19) as f32 / 130.))
            .collect();
        let mut z = vec![0.; context.len()];
        let ends = vec![length - 1; block.len()];
        let outputs = match gather {
            Some("requested-ends") => crate::typed_lookup::Outputs::Positions(&ends),
            _ => crate::typed_lookup::Outputs::All {
                retain_adjoint: training,
            },
        };
        let start = std::time::Instant::now();
        let tape = crate::typed_lookup::Tape::with_outputs(
            &x,
            &context,
            path,
            (cfg.width, length),
            read,
            outputs,
            |v| router.nearest(&layout, store, v),
        );
        tape.add_output(&mut z);
        elapsed += start.elapsed();
        if training {
            let dy: Vec<_> = (0..x.len())
                .map(|i| ((i * 17 % 113) as f32 - 50.) / 67.)
                .collect();
            let mut dx = vec![0.; x.len()];
            let mut dh = dx.clone();
            let mut dw = vec![0.; path.len()];
            let start = std::time::Instant::now();
            tape.backward(&context, path, &dy, &mut dx, &mut dh, &mut dw);
            adjoint += start.elapsed();
            for value in [&z, &dx, &dh, &dw] {
                output.extend(value.iter().copied().flat_map(f32::to_le_bytes));
            }
        } else if gather.is_some() {
            for sequence in z.chunks_exact(cfg.width * length) {
                output.extend(
                    sequence[cfg.width * (length - 1)..]
                        .iter()
                        .copied()
                        .flat_map(f32::to_le_bytes),
                );
            }
        } else {
            output.extend(z.into_iter().flat_map(f32::to_le_bytes));
        }
    }
    let mut result = json!({"step":point["step"],"hops":hops,"passes":passes,"examples":20,"batch":batch,
       "digest":hash(&output),"typed_read_seconds":elapsed.as_secs_f64()});
    if let Some(gather) = gather {
        result["positions"] = json!(gather);
    }
    if training {
        result["backward_seconds"] = json!(adjoint.as_secs_f64());
        result["weights_sha256"] = json!(weight_digest(&w));
        result["workers"] = json!(crate::typed_lookup::worker_limit());
    }
    println!("{result}");
    Ok(())
}
