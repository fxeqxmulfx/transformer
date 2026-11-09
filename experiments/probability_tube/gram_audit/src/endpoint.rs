//! Future evidence is diagnostic and cannot enter the sealed constructor.
use crate::{
    Result,
    application::State,
    classes::{Partition, dot, energy},
    domain::data::{Data, Plan},
    evidence::require,
    gpu,
    infrastructure::engine::Engine,
    metric_analysis, observe, reference, sources,
};
use serde_json::{Value, json};
pub struct Case<'a> {
    pub data: &'a Data,
    pub plan: &'a Plan,
    pub source: [&'a Value; 2],
    pub forecast: &'a Value,
    pub previous: [&'a Value; 2],
    pub old_gpu: &'a gpu::Snapshot,
    pub partition: Partition,
}
fn weighted(x: &[f64], r: &[f64]) -> f64 {
    x.iter().enumerate().map(|(k, x)| r[k % 128] * x * x).sum()
}
fn scaled_excess(a: f64, b: f64) -> f64 {
    (a - b) / (1. + a.abs().max(b.abs()))
}
pub fn check(ctx: &Case<'_>, next: &State, e: &Engine, horizon: usize) -> Result<Value> {
    let time = ctx.forecast["anchor"].as_u64().ok_or("anchor")? as usize + horizon;
    let snapshot = gpu::snapshot(e, ctx.data)?;
    let p: Vec<_> = next.parameters.iter().map(|v| *v as f64).collect();
    let execution = observe::execution(&p, &snapshot, &ctx.data.train)?;
    let logits = crate::algebra::linear(&snapshot.features, 4704, 128, &p[..97 * 128], 97);
    let real_q = observe::probabilities(&logits, &ctx.partition.active);
    let motion: Vec<_> = snapshot
        .features
        .as_chunks::<128>()
        .0
        .iter()
        .zip(ctx.old_gpu.features.as_chunks::<128>().0)
        .map(|(a, b)| {
            a.iter()
                .zip(b)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let current = observe::readout(next);
    let w = ctx.partition.project(&current.p)[1].clone();
    let m = ctx.partition.project(&current.m)[1].clone();
    let s = sources::number(&ctx.forecast["s"])?;
    let metrics = [vec![1.; 128], sources::array(&ctx.forecast["r"])?];
    let mut full = vec![];
    for (metric, r) in metrics.iter().enumerate() {
        let before = sources::number(&ctx.forecast["metrics"][metric]["before"])?;
        let after = metric_analysis::potential(&w, &m, r, s);
        for previous in ctx.previous {
            let endpoint = previous["endpoints"]
                .as_array()
                .ok_or("previous endpoints")?
                .iter()
                .find(|v| v["horizon"] == horizon)
                .ok_or("previous endpoint")?;
            require(endpoint["status"] == "pass", "previous numerical failure")?;
            let recorded = &endpoint["full_potentials"][metric];
            require(
                scaled_excess(after, sources::number(&recorded["after"])?).abs() <= 1e-8
                    && recorded["before"] == before,
                "full potential changed",
            )?;
        }
        full.push(json!({"metric":(["isotropic","feature"][metric]),"before":before,"after":after,
            "change":after-before,"relative_change":(after-before)/before,"increase":scaled_excess(after,before)>1e-8}));
    }
    let mut gradient = None;
    let mut partial_error = 0f64;
    let mut clip = 1.;
    if time < 10000 {
        let actual = gpu::minibatch(e, ctx.data, ctx.plan, time)?;
        let indices = probability_gram::batch_indices(&ctx.plan.order, &ctx.plan.boundaries, time)?;
        for (j, &i) in indices.iter().enumerate() {
            require(
                actual.features[j * 128..(j + 1) * 128]
                    == snapshot.features[i as usize * 128..(i as usize + 1) * 128],
                "minibatch/all-input features differ",
            )?;
        }
        let labels: Vec<_> = actual
            .rows
            .as_chunks::<3>()
            .0
            .iter()
            .map(|r| r[2] as usize)
            .collect();
        let ce = reference::ce(&actual.features, &labels, &current.p, &ctx.partition)?;
        require(
            ce.maximum_feature_energy <= observe::ENERGY,
            "minibatch feature energy",
        )?;
        partial_error = ce
            .gradient
            .iter()
            .zip(&actual.partial)
            .map(|(a, b)| scaled_excess(*a, *b).abs())
            .fold(0., f64::max);
        require(partial_error <= 1e-5, "independent CE gradient")?;
        clip = 1. / actual.norm.max(1.);
        gradient = Some(ctx.partition.project(&ce.gradient)[1].clone());
    }
    let radial = gradient.as_ref().map(|g| dot(&w, g));
    if let (Some(g), Some(d)) = (&gradient, radial) {
        require(
            d >= -1e-8 * (1. + energy(&w).sqrt() * energy(g).sqrt()),
            "negative CE radial product",
        )?;
    }
    let mut checks = vec![];
    let mut pass = true;
    for family in 0..2 {
        let source = ctx.source[family]["forecasts"]
            .as_array()
            .ok_or("source horizons")?
            .iter()
            .find(|v| v["horizon"] == horizon)
            .ok_or("source horizon")?;
        let (q, radii) = observe::converted(source)?;
        let q_excess = real_q
            .iter()
            .zip(&q)
            .map(|(a, b)| a - b)
            .fold(f64::NEG_INFINITY, f64::max);
        let feature_excess = motion
            .iter()
            .zip(&radii)
            .map(|(a, b)| a - b)
            .fold(f64::NEG_INFINITY, f64::max);
        let feature_scaled = motion
            .iter()
            .zip(&radii)
            .map(|(a, b)| scaled_excess(*a, *b))
            .fold(f64::NEG_INFINITY, f64::max);
        pass &= q_excess <= 1e-8 && feature_scaled <= 1e-8;
        let forecast = ctx.forecast["forecasts"]
            .as_array()
            .ok_or("coefficient forecasts")?
            .iter()
            .find(|v| v["horizon"] == horizon && v["family"] == ["original", "refined"][family])
            .ok_or("coefficient horizon")?;
        let mut tracks = vec![];
        for (metric, r) in metrics.iter().enumerate() {
            for method in 0..3 {
                let l = sources::number(&forecast["metrics"][metric]["L"][method])?;
                let stability = sources::number(&forecast["metrics"][metric]["stability"][method])?;
                let mut track = json!({"metric":(["isotropic","feature"][metric]),"method":(["uniform","row_trace","gram"][method]),
                    "L":l,"stability":stability,"condition":stability<=2.,"next_minibatch":if time==10000{"terminal"}else{"available"}});
                if let (Some(g), Some(d)) = (&gradient, radial) {
                    let g2 = weighted(g, r);
                    let excess = scaled_excess(g2, l * d);
                    let gc: Vec<_> = g.iter().map(|g| clip * g).collect();
                    let mr: Vec<_> = m.iter().zip(&gc).map(|(m, g)| 0.9 * m + 0.1 * g).collect();
                    let wr: Vec<_> = w
                        .iter()
                        .zip(&mr)
                        .enumerate()
                        .map(|(k, (w, m))| w - s * r[k % 128] * m)
                        .collect();
                    let before = metric_analysis::potential(&w, &m, r, s);
                    let after = metric_analysis::potential(&wr, &mr, r, s);
                    let increase = scaled_excess(after, before) > 1e-8;
                    pass &= excess <= 1e-8 && (stability > 2. || !increase);
                    track["actual_ce"] = json!({"radial":d,"weighted_gradient_energy":g2,"clip":clip,
                        "clipped_radial":clip*d,"clipped_gradient_energy":clip*clip*g2,
                        "inequality_excess":g2-l*d,"scaled_excess":excess,
                        "reference_change":after-before,"reference_increase":increase});
                }
                tracks.push(track);
            }
        }
        checks.push(json!({"family":(["original","refined"][family]),"probability_excess":q_excess,
            "feature_excess":feature_excess,"feature_scaled_excess":feature_scaled,"tracks":tracks}));
    }
    let previous: Vec<_> = ctx
        .previous
        .iter()
        .map(|p| {
            p["endpoints"]
                .as_array()
                .unwrap()
                .iter()
                .find(|e| e["horizon"] == horizon)
                .unwrap()
                .clone()
        })
        .collect();
    Ok(
        json!({"horizon":horizon,"status":if pass{"pass"}else{"fail"},"execution":execution,
        "real_q":real_q.iter().copied().fold(0.,f64::max),"partial_gradient_error":partial_error,
        "checks":checks,"full_potentials":full,"source_findings":previous}),
    )
}
