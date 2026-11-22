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
    let mut routing = Routing::default();
    routing.refresh(l, w, &products);
    routing
}

pub(super) fn run() -> Result<()> {
    let d = 32;
    let k = 7;
    let l = Layout {
        d,
        slots: k,
        dictionary: 0,
        values: k * d,
        gates: (k + k * k) * d,
        bias: (k + k * k + 4) * d,
    };
    let mut w = vec![0.; l.bias + 4];
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
    let (tape, _) = slot_memory::forward(&l, &w, &route(&l, &w), &x, &h, 3);
    if tape[1].transport != Some(4) || tape[4].constant != Some(2) {
        return Err("structural gradient fixture missed the transported or constant branch".into());
    }
    let mut dw = vec![0.; w.len()];
    let dh = slot_memory::backward(&l, &w, &h, &tape, &g, 3, &mut dw);
    for (name, range) in [
        ("prototypes", 0..l.values),
        ("direct_values", l.values..l.gates),
        ("prefix_gates", l.gates..l.bias),
        ("biases", l.bias..l.bias + 4),
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
        let delta = 0.0002;
        let mut measured = [0.; 2];
        for (i, sign) in [-1., 1.].into_iter().enumerate() {
            let (mut p, mut state) = (w.clone(), h.clone());
            let target = if on_hidden { &mut state } else { &mut p };
            for (z, v) in target[range.clone()].iter_mut().zip(&direction) {
                *z += (sign * delta) as f32 * v;
            }
            let (changed, output) = slot_memory::forward(&l, &p, &route(&l, &p), &x, &state, 3);
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
    initialization()
}

fn initialization() -> Result<()> {
    let mut cfg = config(2)?;
    let seed = 19;
    let plain = layout::initialize_model(&cfg, seed);
    cfg.head = head::Head::TripleTransport;
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
