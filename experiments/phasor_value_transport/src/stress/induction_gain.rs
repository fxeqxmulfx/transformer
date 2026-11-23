//! Check the gradient of the shared induction gain on the actual first
//! supervised minibatch of each recall protocol. Labels enter the outer loss
//! and the independent measurement of alignment with the tied classifier.
use super::*;

#[test]
#[ignore = "requires a CUDA device and the original recall tasks"]
fn first_minibatch_gain() -> Result<()> {
    for name in ["basis-recall-easy", "basis-recall-hard"] {
        let (row, task) = load(name)?;
        let cfg = task_config(&row, &task, true)?;
        let batch = row["case"]["batch"].as_u64().ok_or("missing batch")? as usize;
        let budget = row["case"]["max_updates"]
            .as_u64()
            .ok_or("missing budget")? as usize;
        let train = &task.splits["train"];
        let layout = crate::slot_memory::Layout::new(&cfg);
        let bias = architecture::memory_start(&cfg) + layout.bias + 2;
        for seed in 0..3 {
            let weights = layout::initialize_model(&cfg, seed);
            let mut engine = Engine::new(cfg.clone(), &weights)?;
            let mut training = Training::new(&engine, seed, budget)?;
            training.next_epoch(train.len());
            training.packed.fill(train, &training.epoch[..batch]);
            let pack = &training.packed;
            let rows = batch * pack.length;
            let supervised = pack
                .labels
                .iter()
                .filter(|&&y| (y as usize) < cfg.vocab)
                .count();
            let mean_loss =
                |engine: &mut Engine, parameters: &[f32]| -> Result<f64> {
                    engine.parameters.write(parameters)?;
                    engine.forward(&pack.tokens, &pack.labels, batch, pack.length)?;
                    Ok(engine.losses(rows)?.into_iter().map(f64::from).sum::<f64>()
                        / supervised as f64)
                };
            let initial_loss = mean_loss(&mut engine, &weights)?;
            // Check the hypotheses of the tied gain lemma at each supervised
            // position directly, using the actual normalized input and lookup.
            let normalized = engine.normalized_input_for_test(rows)?;
            let prefix = crate::slot_induction::Prefix::new(normalized, cfg.width, pack.length);
            let lookup = &weights[..cfg.vocab * cfg.width];
            for (i, &target) in pack.labels.iter().enumerate() {
                if target as usize >= cfg.vocab {
                    continue;
                }
                let (_, value) = prefix
                    .read(i)
                    .ok_or("missing dense donor on a supervised query")?;
                let start = target as usize * cfg.width;
                let correct = crate::slot_memory::dot(&lookup[start..][..cfg.width], value);
                let mut responses = lookup
                    .chunks_exact(cfg.width)
                    .map(|word| crate::slot_memory::dot(word, value));
                assert!(responses.clone().all(|v| v <= correct) && responses.any(|v| v < correct));
            }
            engine.backward(batch, pack.length)?;
            let derivative = engine.gradients.read()?[bias];
            let epsilon = 0.001f32;
            let mut displaced = weights.clone();
            displaced[bias] += epsilon;
            let upper = mean_loss(&mut engine, &displaced)?;
            displaced[bias] -= 2. * epsilon;
            let lower = mean_loss(&mut engine, &displaced)?;
            let secant = (upper - lower) / (2. * f64::from(epsilon));
            assert!(derivative < 0. && secant < 0. && upper < initial_loss);
            assert!((secant - f64::from(derivative)).abs() < 0.002 * (1. + secant.abs()));
            engine.parameters.write(&weights)?;
            training.update(&mut engine, train, batch, 0)?;
            assert_eq!(engine.parameters.read()?[bias], weights[bias]);
            training.update(&mut engine, train, batch, 1)?;
            let updated = engine.parameters.read()?[bias];
            assert!(updated > weights[bias]);
            println!(
                "{}",
                json!({"case":name,"seed":seed,"supervised":supervised,
                "bias_gradient":derivative,"secant":secant,"initial_ce":initial_loss,
                "initial_gain_bias":weights[bias],"gain_bias_after_second_update":updated})
            );
        }
    }
    Ok(())
}
