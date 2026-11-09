//! Restore b70811e source, build old caches, freeze bounds, then audit and time.
use crate::{
    application::Backend,
    benchmark,
    bounds::Bound,
    cache_cpu,
    device::Modules,
    domain::{Result, model::W},
    gather_gpu,
    infrastructure::{engine::Engine, storage},
    inputs::{Case, Saved, require},
    io::Transfers,
    measure, quality,
    serving::Service,
};
use kernite_cuda::Cuda;
use serde_json::{Value, json};
use std::{fs::File, io::Write, path::Path, time::Instant};
pub fn run(
    case: &Case<'_>,
    saved: &Saved,
    directory: &Path,
    expected: &str,
    cohort: &Instant,
) -> Result<Value> {
    let loading = Instant::now();
    let cuda = Cuda::new()?;
    let modules = Modules::load(&cuda)?;
    let gather = gather_gpu::kernels::Kernels::load(&cuda)?;
    let io = Transfers::new(&cuda)?;
    cuda.sync()?;
    let module_seconds = loading.elapsed().as_secs_f64();
    let setup = Instant::now();
    let mut engine = Engine::with_modules(
        case.arm,
        &saved.state.parameters,
        case.data,
        case.plan,
        modules,
    )?;
    engine.restore(&saved.state)?;
    cuda.sync()?;
    let model_seconds = setup.elapsed().as_secs_f64();
    let old_version = storage::hash(&storage::state_bytes(case.step, &saved.state));
    let diagnostic = Instant::now();
    let source_checks = measure::validate(&engine, case.arm, &saved.state.parameters, case.data)?;
    let old = measure::snapshot(&engine, case.data, false)?;
    for name in ["train", "validation"] {
        require(
            old.metrics[name]["correct"] == saved.last_report[name]["correct"]
                && (old.metrics[name]["loss"].as_f64().unwrap()
                    - saved.last_report[name]["loss"].as_f64().unwrap())
                .abs()
                    < 1e-12,
            "old checkpoint replay failed",
        )?;
    }
    let stale = engine.weight_cache(&old_version)?;
    cuda.sync()?;
    let old_audit_seconds = diagnostic.elapsed().as_secs_f64();
    let cache_start = Instant::now();
    let pairs: Vec<_> = (0..9409)
        .map(|q| [(q / 97) as u32, (q % 97) as u32])
        .collect();
    let prior: Vec<_> = saved.state.parameters.iter().map(|&x| x as f64).collect();
    let old_graph = cache_cpu::graph(case.arm, &prior, &pairs);
    let mut old_executed = Vec::with_capacity(9409 * W);
    for batch in pairs.chunks(512) {
        engine.inference(batch, None, &old_version, &io)?;
        old_executed.extend(engine.inference_features(batch.len(), &io)?);
    }
    let features = cuda.upload(&old_executed)?;
    cuda.sync()?;
    let old_cache_seconds = cache_start.elapsed().as_secs_f64();
    require(
        old_executed
            .iter()
            .zip(&old.features)
            .all(|(a, b)| a.to_bits() == (*b as f32).to_bits()),
        "old serving features differ from source",
    )?;
    let update = Instant::now();
    let offset = case.plan.boundaries[case.step];
    let batch = case.plan.boundaries[case.step + 1] - offset;
    engine.train_step(offset, batch, case.step + 1)?;
    cuda.sync()?;
    let update_seconds = update.elapsed().as_secs_f64();
    let next = engine.state()?;
    let version = storage::hash(&storage::state_bytes(case.step + 1, &next));
    require(version == expected, "source update differs from task #186")?;
    let weights_start = Instant::now();
    let weights = engine.weight_cache(&version)?;
    cuda.sync()?;
    let weight_seconds = weights_start.elapsed().as_secs_f64();
    // These fail before any launch: no updated-network values have been read.
    require(
        engine
            .inference(&[[0, 0]], Some(&stale), &version, &io)
            .is_err(),
        "stale weight cache was accepted",
    )?;
    require(
        engine
            .inference(&[[97, 0]], Some(&weights), &version, &io)
            .is_err(),
        "invalid operands were accepted",
    )?;
    drop(stale);
    let bounds = [false, true].map(|tight| {
        Bound::prepare(
            case.arm,
            &saved.state.parameters,
            &next.parameters,
            &old_graph.nodes,
            &old_version,
            &version,
            tight,
        )
    });
    let persist = Instant::now();
    let frozen = [
        bounds[0].persist(&directory.join("reuse.bin"))?,
        bounds[1].persist(&directory.join("tight-reuse.bin"))?,
    ];
    let feature_bytes = storage::floats(&old_executed);
    let mut file = File::create_new(directory.join("old-features.bin"))?;
    file.write_all(&feature_bytes)?;
    file.sync_all()?;
    let timing_pairs: Vec<_> = case
        .data
        .validation
        .chunks_exact(3)
        .take(512)
        .map(|r| [r[0], r[1]])
        .collect();
    let frozen = json!({"bounds":frozen,"old_feature_sha256":storage::hash(&feature_bytes),"old_feature_bytes":feature_bytes.len(),
        "from":old_version,"to":version,"timing_operands":timing_pairs,"before_updated_forward":true});
    storage::json_write(&directory.join("frozen.json"), &frozen)?;
    let persistence_seconds = persist.elapsed().as_secs_f64();
    // All updated hidden-network evaluations are below this complete freeze.
    let audit_start = Instant::now();
    let new = measure::snapshot(&engine, case.data, false)?;
    let next64: Vec<_> = next.parameters.iter().map(|&x| x as f64).collect();
    let future = cache_cpu::graph(case.arm, &next64, &pairs);
    let allocated = engine.serving_allocated_bytes();
    let mut service = Service {
        engine: &mut engine,
        version: &version,
        old_version: &old_version,
        weights: &weights,
        features: &features,
        bounds: &bounds,
        gather: &gather,
        io: &io,
    };
    let quality = quality::run(
        &mut service,
        quality::Reference {
            data: case.data,
            old_real: &old_graph.features,
            new_real: &future.features,
            old_executed: &old_executed,
            new: &new,
            next: &next64,
            pairs: &pairs,
            directory,
        },
    )?;
    storage::json_write(&directory.join("quality.json"), &quality)?;
    let audit_seconds = audit_start.elapsed().as_secs_f64();
    let reference: Vec<_> = timing_pairs
        .iter()
        .map(|p| {
            let q = (97 * p[0] + p[1]) as usize;
            let row = &new.logits[q * 97..(q + 1) * 97];
            (1..97).fold(0, |w, c| if row[c] > row[w] { c } else { w })
        })
        .collect();
    let timed = benchmark::run(
        &mut service,
        &timing_pairs,
        &reference,
        cohort,
        &directory.join("timing-progress.json"),
    )?;
    let cost = benchmark::setup_costs(
        &timed,
        model_seconds,
        weight_seconds,
        old_cache_seconds,
        &bounds,
    );
    let cpu_common = old_graph.nodes.len() * std::mem::size_of::<cache_cpu::Node>()
        + saved.state.parameters.len() * 4;
    let cpu_capacity = old_graph.nodes.capacity() * std::mem::size_of::<cache_cpu::Node>()
        + saved.state.parameters.capacity() * 4;
    Ok(
        json!({"passed":quality["passed"]==true&&timed["passed"]==true,"state_sha256":version,"source_state_match":true,
        "source_checks":source_checks,"old_metrics":old.metrics,"new_metrics":new.metrics,"stale_cache_rejected":true,"invalid_operands_rejected":true,
        "frozen":frozen,"quality":quality,"timing":timed,"cost":cost,
        "seconds":{"module_loading":module_seconds,"model_setup":model_seconds,"old_audit":old_audit_seconds,
            "old_cache":old_cache_seconds,"source_update":update_seconds,"weight_cache":weight_seconds,
            "bounds":[bounds[0].seconds,bounds[1].seconds],"gram":bounds[1].gram_seconds,"persistence":persistence_seconds,"audit":audit_seconds},
        "bytes":{"source_gpu":allocated,"cached_weight_delta_gpu":weights.bytes(),"reuse_delta_gpu":weights.bytes()+features.len()*4,
            "reuse_cpu_payload":[cpu_common+bounds[0].bytes(),cpu_common+bounds[1].bytes()],
            "reuse_cpu_capacity":[cpu_capacity+bounds[0].capacity(),cpu_capacity+bounds[1].capacity()],
            "audit_arrays_excluded":true,"temporary_build_arrays_excluded":true}}),
    )
}
