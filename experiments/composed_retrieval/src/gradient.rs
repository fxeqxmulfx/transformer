//! Reverse differentiation of ordinary full-vocabulary CE for Mini.
//! Forward-only training queries a separate implementation path in train.rs.
use crate::{
    model::{Model, WIDTH, Wave},
    operator::{Pass, Tables, softmax},
};

fn weighted_sum(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}
struct Routing<'a> {
    model: &'a Model,
    tables: &'a Tables,
    memory: &'a [usize],
}
impl Routing<'_> {
    fn kernel_backward(&self, o: usize, v: usize, derivative: f64, grad: &mut [f64]) {
        if o == v {
            return;
        }
        let (a, b) = (&self.tables.keys[o], &self.tables.keys[v]);
        let f = derivative / WIDTH as f64;
        let start = self.tables.layout.key.start;
        for d in 0..WIDTH {
            let delta = f * (-a.im[d] * b.re[d] + a.re[d] * b.im[d]);
            grad[start + o * WIDTH + d] += delta;
            grad[start + v * WIDTH + d] -= delta;
        }
    }
    fn key_backward(
        &self,
        query: (usize, Option<usize>),
        row: &[f64],
        up: &[f64],
        weight: f64,
        grad: &mut [f64],
    ) {
        let (m, tab, x) = (self.model, self.tables, self.memory);
        let (o, previous_query) = query;
        let l = &tab.layout;
        let mean = weighted_sum(row, up);
        for (j, &v) in x.iter().enumerate() {
            let ds = weight * row[j] * (up[j] - mean);
            grad[l.scale] += ds * tab.similarity[o * m.vocabulary + v];
            grad[l.recency] += ds * j as f64;
            grad[l.bias.start + if j == 0 { m.vocabulary } else { x[j - 1] }] += ds;
            self.kernel_backward(o, v, ds * m.weights[l.scale], grad);
            if let (Some(i), Some(u), Some(&v)) = (l.predecessor, previous_query, x[..j].last()) {
                grad[i] += ds * tab.similarity[u * m.vocabulary + v];
                self.kernel_backward(u, v, ds * m.weights[i], grad);
            }
        }
    }
}
fn route_backward(
    m: &Model,
    tab: &Tables,
    pass: &Pass,
    x: &[usize],
    q: usize,
    read_adjoints: (&[f64], &Wave),
    g: &mut [f64],
) {
    let (first_scores, d2) = read_adjoints;
    let routing = Routing {
        model: m,
        tables: tab,
        memory: x,
    };
    let n = x.len();
    let l = &tab.layout;
    let links = &tab.links[n];
    let dm2: Vec<_> = x.iter().map(|&id| d2.dot(&tab.values[id])).collect();
    let dkeys: Vec<_> = links
        .chunks_exact(n)
        .map(|row| weighted_sum(row, &dm2))
        .collect();
    let mut dm1 = Vec::with_capacity(n);
    for (k, (&id, a)) in x.iter().zip(pass.attention.chunks_exact(n)).enumerate() {
        dm1.push(first_scores[k] + weighted_sum(a, &dkeys));
        routing.key_backward((id, x[..k].last().copied()), a, &dkeys, pass.one[k], g);
    }
    let dquery: Vec<_> = links
        .chunks_exact(n)
        .map(|row| weighted_sum(row, &dm1))
        .collect();
    let aq = &pass.attention[n * n..];
    routing.key_backward((q, x.last().copied()), aq, &dquery, 1., g);
    for (j, row) in links.chunks_exact(n).enumerate() {
        let adj = |k: usize| pass.second_keys[j] * dm2[k] + aq[j] * dm1[k];
        let mean: f64 = row.iter().enumerate().map(|(k, &b)| b * adj(k)).sum();
        for (k, &b) in row.iter().enumerate() {
            l.add_link_gradient(g, j, k, b * (adj(k) - mean));
        }
    }
}
pub fn accumulate(m: &Model, tab: &Tables, prefix: &[usize], label: usize, g: &mut [f64]) {
    let l = &tab.layout;
    let pass = tab.pass(m, prefix);
    let mut residual = tab.logits(m, &pass.hidden);
    softmax(&mut residual);
    residual[label] -= 1.;
    for (entry, r) in g[l.output_offsets.clone()].iter_mut().zip(&residual) {
        *entry += r;
    }
    let mut dh = Wave::default();
    let out = m.capacity / WIDTH as f64;
    for (c, (&r, v)) in residual.iter().zip(&tab.values).enumerate() {
        dh.add(v, out * r);
        let i = l.value.start + c * WIDTH;
        v.angle_gradient(&pass.hidden, &mut g[i..i + WIDTH], out * r);
    }
    let Some((&q, x)) = prefix.split_last() else {
        tab.start.angle_gradient(&dh, &mut g[l.start.clone()], 1.);
        return;
    };
    let turn = &tab.phases[q];
    let at = l.phase.start + q * WIDTH;
    if x.is_empty() {
        pass.hidden.angle_gradient(&dh, &mut g[l.start.clone()], 1.);
        pass.hidden.angle_gradient(&dh, &mut g[at..at + WIDTH], 1.);
        return;
    }
    let h = m.weights[l.depth];
    let u = m.weights[l.product];
    let rotated = pass.product_read.times(turn, false);
    let query_radius = tab.radius(m, q);
    let base = Wave::mix(&pass.s1, &pass.s2, h);
    g[l.product] += query_radius * dh.dot(&rotated) - dh.dot(&base);
    g[l.depth] += (1. - u) * (dh.dot(&pass.s2) - dh.dot(&pass.s1));
    rotated.angle_gradient(&dh, &mut g[at..at + WIDTH], u * query_radius);
    if m.learned_product_radii {
        g[l.radii.start + q] += u * dh.dot(&rotated);
    }
    let (mut plain, mut d2) = (Wave::default(), Wave::default());
    plain.add(&dh, (1. - u) * (1. - h));
    let product_adjoint = dh.times(turn, true);
    d2.add(&dh, (1. - u) * h);
    let mut first_scores = Vec::with_capacity(x.len());
    for (k, &id) in x.iter().enumerate() {
        let mut d1 = plain;
        d1.add(&product_adjoint, u * query_radius * tab.radius(m, id));
        if m.learned_product_radii {
            g[l.radii.start + id] +=
                pass.one[k] * u * query_radius * product_adjoint.dot(&tab.values[id]);
        }
        if x.len() > 1 {
            first_scores.push(d1.dot(&tab.values[id]));
        }
        let start = l.value.start + id * WIDTH;
        let row = &mut g[start..start + WIDTH];
        tab.values[id].angle_gradient(&d1, row, pass.one[k]);
        tab.values[id].angle_gradient(&d2, row, pass.two[k]);
    }
    if x.len() > 1 {
        route_backward(m, tab, &pass, x, q, (&first_scores, &d2), g);
    }
}
