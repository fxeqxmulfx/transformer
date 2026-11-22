//! Directional derivatives with the constant and triple reads both active.
use super::*;
use crate::slot_memory::{self, Layout, Routing};

fn route(l: &Layout, w: &[f32]) -> Routing {
    let mut products = Vec::with_capacity(l.slots.pow(3));
    for m in w[l.values..l.gates].chunks_exact(l.d) {
        for i in 0..l.slots {
            products.push(slot_memory::dot(m, l.prototype(w, i)));
        }
    }
    let mut routing = Routing::new(architecture::HEAD);
    routing.refresh(l, w, &products);
    routing
}

pub(super) fn run() -> Result<()> {
    crate::slot_bank_checks::run()?;
    check_addresses()?;
    // The actual FP32 mixture can lose the smallest secant to rounding.
    // Both perturbation sizes must agree and keep every discrete read fixed.
    mixture(0.001)?;
    mixture(0.002)?;
    initialization()?;
    fixed_evaluation()
}

fn fixed_evaluation() -> Result<()> {
    let mut cfg = config(2)?;
    cfg.head = architecture::HEAD;
    let mut weights = layout::initialize_model(&cfg, 13);
    crate::slot_checks::fixture(&cfg, &mut weights);
    let mut muted = weights.clone();
    crate::slot_checks::disable_reads(&cfg, &mut muted, &[0, 1, 2, 3]);
    let input = &weights[..2 * cfg.width];
    let mut e = Engine::new(cfg.clone(), &weights)?;
    let reference = e.infer_embeddings(input, 1, 2)?;
    let prefix = e.infer_embeddings(&input[..cfg.width], 1, 1)?;
    let interrupted: Result<()> = e.with_fixed_slot_weights(|e| {
        let actual = e.infer_embeddings(input, 1, 2)?;
        let shorter = e.infer_embeddings(&input[..cfg.width], 1, 1)?;
        if actual.logits != reference.logits
            || actual.hidden != reference.hidden
            || shorter.logits != prefix.logits
            || shorter.hidden != prefix.hidden
        {
            return Err("fixed-weight prefix differs from ordinary inference".into());
        }
        Err("end of cache fixture".into())
    });
    if !matches!(interrupted, Err(ref err) if err.to_string() == "end of cache fixture") {
        return Err("fixed-weight error path failed".into());
    }
    e.parameters.write(&muted)?;
    let after = e.infer_embeddings(input, 1, 2)?;
    let expected = Engine::new(cfg, &muted)?.infer_embeddings(input, 1, 2)?;
    if after.logits == reference.logits
        || after.logits != expected.logits
        || after.hidden != expected.hidden
    {
        return Err("changed weights after scoring did not match fresh inference".into());
    }
    e.parameters.write(&weights)?;
    let restored = e.with_fixed_slot_weights(|e| e.infer_embeddings(input, 1, 2))?;
    if restored.logits != reference.logits || restored.hidden != reference.hidden {
        return Err("new scoring scope reused a previous set of weights".into());
    }
    println!(
        "fixed scoring prefixes=2 weights=2 matched_logits={}",
        reference.logits.len()
    );
    Ok(())
}

fn mixture(delta: f64) -> Result<()> {
    let d = 32;
    let k = 7;
    let l = Layout {
        d,
        slots: k,
        gate_mode: architecture::HEAD.read_gates(),
        gate_scales: slot_memory::scales(d, architecture::HEAD.unit_read_gates()),
        dictionary: 0,
        values: k * d,
        gates: (k + k * k) * d,
        bias: (k + k * k + architecture::HEAD.read_gates().count()) * d,
    };
    let mut w = vec![0.; l.end()];
    let mut rng = random::Random::new(4147);
    for i in 0..k {
        for j in 0..d {
            w[i * d + j] = if (i & j).count_ones().is_multiple_of(2) {
                1.
            } else {
                -1.
            };
        }
    }
    // Two generators act on seven unnamed dense vectors. The third partial
    // map has only two supports, and the fourth has three equal images.
    for (a, b, y) in (0..k)
        .map(|b| (0, b, (b + 1) % k))
        .chain((0..k).map(|b| (1, b, (3 * b) % k)))
        .chain((0..k).map(|b| (4, b, b)))
        .chain([(2, 0, 3), (2, 1, 5), (3, 0, 2), (3, 1, 2), (3, 2, 2)])
    {
        for j in 0..d {
            w[l.values + (a * k + b) * d + j] = 0.3 * w[y * d + j];
        }
    }
    for v in &mut w[l.gates..] {
        *v = rng.gauss(0.12, 0.07) as f32;
    }
    let mut x = Vec::new();
    for i in [2, 4, 5, 3, 4, 1, 0, 1, 6] {
        for v in l.prototype(&w, i) {
            x.push(v + rng.gauss(0., 0.25) as f32);
        }
    }
    let h: Vec<_> = (0..x.len()).map(|_| rng.gauss(0., 0.4) as f32).collect();
    let g: Vec<_> = (0..x.len()).map(|_| rng.gauss(0., 0.17) as f32).collect();
    let routing = route(&l, &w);
    let cells = slot_memory::scalar_cells(&l, &w, &routing, &x, &h, 3);
    let (tape, _) = slot_memory::forward(&l, &w, &routing, &cells, &h, 3);
    if tape[1].transport != Some(4) || tape[4].constant != Some(2) {
        return Err("structural gradient fixture missed the transported or constant branch".into());
    }
    let (dh, adjoint) = slot_memory::backward(&l, &w, &h, &tape, &g, 3);
    let dw = adjoint.dense(&l);
    for (name, range) in [
        ("prototypes", 0..l.values),
        ("direct_values", l.values..l.gates),
        ("prefix_gates", l.gates..l.bias),
        ("biases", l.bias..l.end()),
        ("prefix_hidden", 0..h.len()),
    ] {
        let on_hidden = name == "prefix_hidden";
        let gradient = if on_hidden { &dh } else { &dw };
        let direction: Vec<_> = range.clone().map(|_| rng.gauss(0., 0.3) as f32).collect();
        let analytic: f64 = direction
            .iter()
            .zip(&gradient[range.clone()])
            .map(|(&v, &g)| f64::from(v) * f64::from(g))
            .sum();
        let mut measured = [0.; 2];
        for (i, sign) in [-1., 1.].into_iter().enumerate() {
            let (mut p, mut state) = (w.clone(), h.clone());
            let target = if on_hidden { &mut state } else { &mut p };
            for (z, v) in target[range.clone()].iter_mut().zip(&direction) {
                *z += (sign * delta) as f32 * v;
            }
            let routing = route(&l, &p);
            let cells = slot_memory::scalar_cells(&l, &p, &routing, &x, &state, 3);
            let (changed, output) = slot_memory::forward(&l, &p, &routing, &cells, &state, 3);
            for (old, new) in tape.iter().zip(changed) {
                if (old.cell, old.constant, old.transport)
                    != (new.cell, new.constant, new.transport)
                {
                    return Err(
                        format!("gradient fixture crossed a {name} routing boundary").into(),
                    );
                }
            }
            measured[i] = output
                .iter()
                .zip(&g)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>();
        }
        let numeric = (measured[1] - measured[0]) / (2. * delta);
        if analytic.abs() < 1e-5
            || !numeric.is_finite()
            || (analytic - numeric).abs() > 0.004 * (1. + analytic.abs())
        {
            return Err(
                format!("slot mixture {name}: analytic={analytic} numeric={numeric}").into(),
            );
        }
        println!("slot mixture {name} analytic={analytic:.7} numeric={numeric:.7}");
    }
    Ok(())
}

// Compare the real device kernel with the carried scalar search on arbitrary
// dense inputs, duplicate directions, null vectors and separate sequences.
fn check_addresses() -> Result<()> {
    let cuda = kernite_cuda::Cuda::new()?;
    let kernels = crate::slot_addresses::kernels::Kernels::load(&cuda)?;
    let mut rng = random::Random::new(7293);
    for (d, length, vocab) in [
        (7, 1, 5),
        (7, 9, 11),
        (64, 11, 37),
        (128, 5, 193),
        (256, 4, 320),
    ] {
        let mut cfg = config(3)?;
        cfg.head = architecture::HEAD;
        cfg.width = d;
        cfg.vocab = vocab;
        let l = Layout::new(&cfg);
        let rows = 3 * length;
        let mut w = vec![0.; l.end()];
        for coordinate in &mut w[..l.values] {
            *coordinate = rng.gauss(0., 1.) as f32;
        }
        w.copy_within(0..d, d);
        w[2 * d..3 * d].fill(0.);
        let mut x: Vec<_> = (0..rows * d).map(|_| rng.gauss(0., 1.) as f32).collect();
        let mut h: Vec<_> = (0..rows * d).map(|_| rng.gauss(0., 0.25) as f32).collect();
        for row in (0..rows).step_by(4) {
            x[row * d..(row + 1) * d].copy_from_slice(&w[..d]);
            h[row * d..(row + 1) * d].fill(0.);
        }
        x[d..2 * d].fill(0.);
        let mut routing = Routing::new(architecture::HEAD);
        routing.refresh(&l, &w, &vec![0.; l.slots.pow(3)]);
        let expected = slot_memory::scalar_cells(&l, &w, &routing, &x, &h, length);
        let transposed: Vec<_> = (0..l.slots * d)
            .map(|i| w[i % l.slots * d + i / l.slots])
            .collect();
        let (input, hidden) = (cuda.upload(&x)?, cuda.upload(&h)?);
        let dictionary = cuda.upload(&transposed)?;
        let inverse = cuda.upload(routing.inverse_norms())?;
        let cells = cuda.alloc::<u32>(rows)?;
        unsafe {
            kernels.cells(
                kernite_cuda::Grid::new(rows as u32, l.slots as u32),
                cells.ptr(),
                input.ptr(),
                hidden.ptr(),
                dictionary.ptr(),
                inverse.ptr(),
                d as u32,
                length as u32,
                (d as f32).sqrt().recip(),
            )?;
        }
        let measured = cells.read()?;
        if measured != expected
            || length > 1 && measured.iter().all(|&v| v == crate::slot_orbits::ABSENT)
        {
            return Err(format!(
                "parallel slot addresses differ from the scalar scan at width {d}, length {length}"
            )
            .into());
        }
        println!("parallel slot addresses width={d} length={length} scalar_matches={rows}");
    }
    Ok(())
}

fn initialization() -> Result<()> {
    let mut cfg = config(2)?;
    let seed = 19;
    let plain = layout::initialize_model(&cfg, seed);
    cfg.head = architecture::HEAD;
    let w = layout::initialize_model(&cfg, seed);
    let external = cfg.width * cfg.vocab;
    if w[..external] != plain[..external] {
        return Err("slot initialization changed the pinned random lookup".into());
    }
    let mut renamed = w.clone();
    let mut order: Vec<_> = (0..cfg.vocab).collect();
    random::Random::new(901).shuffle(&mut order);
    for (i, &j) in order.iter().enumerate() {
        renamed[j * cfg.width..(j + 1) * cfg.width]
            .copy_from_slice(&plain[i * cfg.width..(i + 1) * cfg.width]);
    }
    slot_memory::initialize(&cfg, seed, &mut renamed);
    let memory = architecture::memory_start(&cfg);
    let end = architecture::start(&cfg) + cfg.head.parameters(&cfg);
    if w[memory..end] != renamed[memory..end] {
        return Err("vocabulary order affected the initial hidden memory".into());
    }
    println!(
        "slot initialization random_lookup_values={external} invariant_internal_weights={}",
        end - memory
    );
    Ok(())
}
