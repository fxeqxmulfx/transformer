//! Dense value memory and learned gates for the triple-transport read.
//! Its addresses are nearest directions in a trainable, unordered dictionary.
use crate::{
    Config, architecture,
    random::Random,
    slot_orbits::{self, ABSENT},
};

pub struct Layout {
    pub d: usize,
    pub slots: usize,
    pub gate_mode: crate::head::ReadGates,
    pub gate_scales: [f32; 4],
    pub dictionary: usize,
    pub values: usize,
    pub gates: usize,
    pub bias: usize,
}

pub fn capacity(head: crate::head::Head, width: usize, vocab: usize) -> usize {
    const LIMIT: usize = 256;
    if head.uses_lookup_capacity() && (head.peer_read().is_none() || vocab <= LIMIT) {
        vocab.min(LIMIT)
    } else {
        width.min(128)
    }
}

pub fn parameter_count(head: crate::head::Head, d: usize, k: usize) -> usize {
    let phase = if head.has_triple_phase() {
        3 * crate::head::MEMORY_WIDTH * d
    } else {
        0
    };
    let gates = head.read_gates().count();
    4 * d + (k + k * k + gates) * d + gates + phase
}

pub fn phase_start(cfg: &Config) -> usize {
    architecture::memory_start(cfg) + Layout::new(cfg).end()
}

pub fn phase_outputs(cfg: &Config) -> std::ops::Range<usize> {
    let channels = crate::head::MEMORY_WIDTH * cfg.width;
    let start = phase_start(cfg) + channels;
    start..start + 2 * channels
}

impl Layout {
    pub fn new(cfg: &Config) -> Self {
        let d = cfg.width;
        let slots = capacity(cfg.head, d, cfg.vocab);
        let dictionary = 0;
        let values = dictionary + slots * d;
        let gates = values + slots * slots * d;
        let gate_mode = cfg.head.read_gates();
        Self {
            d,
            slots,
            gate_mode,
            gate_scales: scales(d, cfg.head.unit_read_gates()),
            dictionary,
            values,
            gates,
            bias: gates + gate_mode.count() * d,
        }
    }

    pub fn end(&self) -> usize {
        self.bias + self.gate_mode.count()
    }

    pub fn prototype<'a>(&self, w: &'a [f32], slot: usize) -> &'a [f32] {
        let start = self.dictionary + slot * self.d;
        &w[start..start + self.d]
    }

    pub fn row<'a>(&self, w: &'a [f32], cell: usize) -> &'a [f32] {
        let start = self.values + cell * self.d;
        &w[start..start + self.d]
    }
}

fn vector_order(a: &[f32], b: &[f32]) -> std::cmp::Ordering {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.total_cmp(y))
        .find(|o| !o.is_eq())
        .unwrap_or(std::cmp::Ordering::Equal)
}

pub fn initialize(cfg: &Config, seed: u32, w: &mut [f32]) {
    let l = Layout::new(cfg);
    let mut rng = Random::new(seed.wrapping_add(30_000));
    let mut directions: Vec<_> = w[..cfg.vocab * l.d]
        .chunks_exact(l.d)
        .map(<[f32]>::to_vec)
        .collect();
    directions.sort_by(|a, b| vector_order(a, b));
    directions.truncate(l.slots);
    while directions.len() < l.slots {
        directions.push((0..l.d).map(|_| rng.gauss(0., 0.02) as f32).collect());
    }
    for a in &mut directions {
        let scale = (l.d as f32).sqrt() * inverse_norm(a);
        for x in a {
            *x *= scale;
        }
    }
    directions.sort_by(|a, b| vector_order(a, b));
    let w = &mut w[architecture::memory_start(cfg)..];
    for (dst, a) in w[l.dictionary..l.values]
        .chunks_exact_mut(l.d)
        .zip(directions)
    {
        dst.copy_from_slice(&a);
    }
    w[l.values..l.end()].fill(0.);
    if cfg.head.has_triple_phase() {
        let channels = crate::head::MEMORY_WIDTH * l.d;
        let phase = &mut w[l.end()..l.end() + 3 * channels];
        let (increments, readout) = phase.split_at_mut(channels);
        let mut rng = Random::new(seed.wrapping_add(30_000));
        let scale = (l.d as f64).sqrt().recip();
        for weight in increments {
            *weight = rng.gauss(0., scale) as f32;
        }
        readout.fill(0.);
    }
}

pub fn dot(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

fn inverse_norm(v: &[f32]) -> f32 {
    let n = dot(v, v).sqrt();
    if n > 0. { n.recip() } else { 0. }
}

#[derive(Default)]
pub struct Routing {
    bank: Vec<u32>,
    priority: Vec<usize>,
    reads: Option<slot_orbits::Reads>,
    graph: slot_orbits::TripleGraph,
    peer_search: Option<crate::head::PeerSearch>,
    pairs: Option<slot_orbits::PairRead>,
    pub(crate) mute_proposals: [bool; 2],
    pub(crate) mute_context: bool,
    inverse: Vec<f32>,
}

pub struct BankGeometry {
    pub priority: Vec<usize>,
    pub value_scales: Vec<f32>,
}

impl Routing {
    pub fn new(head: crate::head::Head) -> Self {
        Self {
            graph: slot_orbits::TripleGraph::new(head.separates_triple_roles()),
            peer_search: head.peer_search(),
            pairs: head.peer_read().and_then(|r| r.pairs),
            ..Self::default()
        }
    }

    pub fn inverse_norms(&self) -> &[f32] {
        &self.inverse
    }

    pub fn geometry(&mut self, l: &Layout, w: &[f32]) -> BankGeometry {
        self.inverse = (0..l.slots)
            .map(|i| inverse_norm(l.prototype(w, i)))
            .collect();
        let mut priority: Vec<_> = (0..l.slots).collect();
        priority.sort_by(|&a, &b| vector_order(l.prototype(w, a), l.prototype(w, b)));
        // Coincident prototypes have no content-based priority. Their cells
        // cannot be anchors, including when a tie straddles the anchor limit.
        let mut distinct = vec![true; l.slots];
        for pair in priority.windows(2) {
            if l.prototype(w, pair[0]) == l.prototype(w, pair[1]) {
                distinct[pair[0]] = false;
                distinct[pair[1]] = false;
            }
        }
        let damping = l.d as f32 * f32::EPSILON;
        let value_scales = w[l.values..l.gates]
            .chunks_exact(l.d)
            .enumerate()
            .map(|(cell, v)| {
                if !distinct[cell / l.slots] || !distinct[cell % l.slots] {
                    return 0.;
                }
                let energy = dot(v, v);
                if energy <= damping {
                    0.
                } else {
                    (energy + damping).sqrt().recip()
                }
            })
            .collect();
        BankGeometry {
            priority,
            value_scales,
        }
    }

    pub fn install(&mut self, l: &Layout, bank: Vec<u32>, priority: Vec<usize>) {
        if self.bank != bank || self.priority != priority {
            let mut reads = self.graph.complete(l.slots, &bank, &priority);
            match self.peer_search {
                Some(crate::head::PeerSearch::Aligned) => {
                    slot_orbits::fill_from_peers(l.slots, &bank, &mut reads.transport)
                }
                Some(crate::head::PeerSearch::Witnesses) => slot_orbits::fill_from_witnesses(
                    l.slots,
                    &bank,
                    &priority,
                    &reads.constant,
                    &mut reads.transport,
                ),
                Some(crate::head::PeerSearch::Iterated(roles)) => slot_orbits::fill_in_rounds(
                    l.slots,
                    &bank,
                    &priority,
                    &reads.constant,
                    &mut reads.transport,
                    roles,
                ),
                None => {}
            }
            if let Some(pairs) = self.pairs {
                slot_orbits::fill_from_pair_classes(
                    l.slots,
                    &bank,
                    &priority,
                    &reads.constant,
                    &mut reads.transport,
                    pairs,
                );
            }
            self.reads = Some(reads);
            self.bank = bank;
            self.priority = priority;
        }
    }

    /// Scalar reference used by the gradient and device equivalence checks.
    pub fn refresh(&mut self, l: &Layout, w: &[f32], products: &[f32]) {
        let geometry = self.geometry(l, w);
        let bank = products
            .chunks_exact(l.slots)
            .zip(&geometry.value_scales)
            .map(|(scores, &scale)| trusted_slot(scores, &self.inverse, scale))
            .collect();
        self.install(l, bank, geometry.priority);
    }

    fn nearest(&self, l: &Layout, w: &[f32], v: &[f32]) -> Option<usize> {
        let mut best = 0.;
        let mut winner = None;
        for (i, &inv) in self.inverse.iter().enumerate() {
            let similarity = dot(v, l.prototype(w, i)) * inv;
            if similarity > best {
                best = similarity;
                winner = Some(i);
            } else if similarity == best {
                winner = None;
            }
        }
        winner
    }
}

pub fn trusted_slot(scores: &[f32], inverse: &[f32], scale: f32) -> u32 {
    if scale == 0. {
        return ABSENT;
    }
    let mut best = std::f32::consts::FRAC_1_SQRT_2;
    let mut winner = ABSENT;
    for (i, (&score, &inv)) in scores.iter().zip(inverse).enumerate() {
        let cosine = score * inv * scale;
        if cosine > best {
            best = cosine;
            winner = i as u32;
        } else if cosine == best {
            winner = ABSENT;
        }
    }
    winner
}

#[derive(Clone, Default)]
pub struct Row {
    pub cell: Option<usize>,
    pub constant: Option<usize>,
    pub transport: Option<usize>,
    pub coefficients: [f32; 4],
}

pub fn scales(d: usize, unit_gates: bool) -> [f32; 4] {
    let root = (d as f32).sqrt();
    if unit_gates {
        [1., 1., root, root]
    } else {
        [root.recip(), d as f32, root, root]
    }
}

impl Row {
    fn structural(&self, branch: usize, shared: bool) -> Option<usize> {
        match branch {
            2 if shared => self.constant.or(self.transport),
            2 => self.constant,
            3 => self.transport,
            _ => unreachable!(),
        }
    }

    pub fn read<'a>(
        &self,
        branch: usize,
        l: &Layout,
        w: &'a [f32],
        h: &'a [f32],
        prefix_value: Option<&'a [f32]>,
    ) -> Option<&'a [f32]> {
        match branch {
            0 => Some(h),
            1 => self.cell.map(|i| l.row(w, i)),
            2 if prefix_value.is_some() => prefix_value,
            2 | 3 => self
                .structural(branch, l.gate_mode.shared())
                .map(|i| l.prototype(w, i)),
            _ => unreachable!(),
        }
    }
}

/// Independent scalar scan used to verify batched device addressing.
pub fn scalar_cells(
    l: &Layout,
    w: &[f32],
    route: &Routing,
    x: &[f32],
    hidden: &[f32],
    length: usize,
) -> Vec<u32> {
    let d = l.d;
    let units = (d as f32).sqrt().recip();
    let mut key = vec![0.; d];
    let mut prior = None;
    let mut cells = Vec::with_capacity(x.len() / d);
    for (r, (q, h)) in x.chunks_exact(d).zip(hidden.chunks_exact(d)).enumerate() {
        let cell = if r.is_multiple_of(length) {
            None
        } else {
            prior
                .zip(route.nearest(l, w, q))
                .map(|(a, b)| (a * l.slots + b) as u32)
        };
        cells.push(cell.unwrap_or(ABSENT));
        if !(r + 1).is_multiple_of(length) {
            for j in 0..d {
                key[j] = q[j] + units * h[j];
            }
            prior = route.nearest(l, w, &key);
        }
    }
    cells
}

pub fn forward(
    l: &Layout,
    w: &[f32],
    route: &Routing,
    cells: &[u32],
    hidden: &[f32],
    length: usize,
    prefix: Option<&crate::slot_induction::Prefix>,
) -> (Vec<Row>, Vec<f32>) {
    let d = l.d;
    assert_eq!(cells.len() * d, hidden.len());
    let scale = l.gate_scales;
    let units = (d as f32).sqrt().recip();
    let mut tape = Vec::with_capacity(cells.len());
    let mut out = vec![0.; hidden.len()];
    for (r, ((&cell, h), dst)) in cells
        .iter()
        .zip(hidden.chunks_exact(d))
        .zip(out.chunks_exact_mut(d))
        .enumerate()
    {
        let mut row = Row {
            coefficients: [scale[0], scale[1], 0., 0.],
            ..Row::default()
        };
        let position = l.gate_mode.position(r, length);
        if route.mute_context {
            row.coefficients[0] = 0.;
        }
        let first = l.gate_mode.first_learned();
        for (i, g) in row.coefficients[first..l.gate_mode.reads()]
            .iter_mut()
            .enumerate()
        {
            let mut value = w[l.bias + i];
            if let Some(position) = position {
                let weight = l.gates + i * d;
                value += units * dot(&w[weight..weight + d], &hidden[position * d..][..d]);
            }
            *g += scale[first + i] * value;
        }
        if cell != ABSENT {
            let cell = cell as usize;
            let reads = route.reads.as_ref().expect("routing was refreshed");
            row.cell = Some(cell);
            row.constant = (!route.mute_proposals[0] && reads.constant[cell] != ABSENT)
                .then_some(reads.constant[cell] as usize);
            row.transport = (!route.mute_proposals[1] && reads.transport[cell] != ABSENT)
                .then_some(reads.transport[cell] as usize);
        }
        let prior = prefix.and_then(|p| p.read(r)).map(|(_, value)| value);
        if prior.is_some() {
            row.constant = None;
            row.transport = None;
        }
        for (i, &c) in row.coefficients[..l.gate_mode.reads()].iter().enumerate() {
            if let Some(value) = row.read(i, l, w, h, prior) {
                for (y, &v) in dst.iter_mut().zip(value) {
                    *y += c * v;
                }
            }
        }
        tape.push(row);
    }
    (tape, out)
}

pub struct Backward {
    pub hidden: Vec<f32>,
    pub parameters: crate::slot_adjoint::Adjoint,
    pub input: Vec<f32>,
}

pub fn backward(
    l: &Layout,
    w: &[f32],
    hidden: &[f32],
    rows: &[Row],
    dy: &[f32],
    length: usize,
    prefix: Option<&crate::slot_induction::Prefix>,
) -> Backward {
    let d = l.d;
    let scale = l.gate_scales;
    let units = (d as f32).sqrt().recip();
    let mut dh = vec![0.; rows.len() * d];
    let mut input = if prefix.is_some() {
        vec![0.; rows.len() * d]
    } else {
        vec![]
    };
    let mut adjoint = crate::slot_adjoint::Adjoint::new(l, rows);
    for (r, row) in rows.iter().enumerate() {
        let v = &dy[r * d..(r + 1) * d];
        let h = &hidden[r * d..(r + 1) * d];
        let prior = prefix.and_then(|p| p.read(r));
        let prefix_value = prior.map(|(_, value)| value);
        for (i, &c) in row.coefficients[..l.gate_mode.reads()].iter().enumerate() {
            let Some(value) = row.read(i, l, w, h, prefix_value) else {
                continue;
            };
            if let Some(gate) = i.checked_sub(l.gate_mode.first_learned()) {
                let dg = scale[i] * dot(value, v);
                adjoint.gates[l.bias - l.gates + gate] += dg;
                if let Some(position) = l.gate_mode.position(r, length) {
                    for j in 0..d {
                        let k = l.gates + gate * d + j;
                        adjoint.gates[k - l.gates] += units * dg * hidden[position * d + j];
                        dh[position * d + j] += units * dg * w[k];
                    }
                }
            }
            let dst = match (i, prior) {
                (0, _) => &mut dh[r * d..(r + 1) * d],
                (1, _) => adjoint.row(l.slots + row.cell.unwrap(), d),
                (2, Some((source, _))) => &mut input[source * d..][..d],
                (2 | 3, _) => {
                    let slot = row.structural(i, l.gate_mode.shared()).unwrap();
                    adjoint.row(slot, d)
                }
                _ => unreachable!(),
            };
            for (g, &z) in dst.iter_mut().zip(v) {
                *g += c * z;
            }
        }
    }
    Backward {
        hidden: dh,
        parameters: adjoint,
        input,
    }
}
#[cfg(test)]
mod capacity_tests {
    use super::capacity;
    use crate::head::{Head, PeerRead, PeerSearch};

    #[test]
    fn removing_search_preserves_dictionary_and_unit_gates() {
        for full in [
            PeerRead::UNIT_GRAPH,
            PeerRead::SHARED_GRAPH,
            PeerRead::CURRENT_GRAPH,
        ] {
            let removed = PeerRead {
                search: None,
                ..full
            };
            for read in [full, removed] {
                for head in [Head::PeerTriple(read), Head::PeerTriplePhase(read)] {
                    assert_eq!(head.read_gates(), full.gates);
                    assert_eq!(super::scales(64, head.unit_read_gates()), [1., 1., 8., 8.]);
                    assert_eq!(capacity(head, 64, 97), 97);
                    assert_eq!(capacity(head, 64, 300), 64);
                }
            }
        }
        let original = Head::peer_phase(PeerRead::UNIT_GRAPH.search.unwrap());
        assert_eq!(
            super::scales(64, original.unit_read_gates()),
            [0.125, 64., 8., 8.]
        );
    }

    #[test]
    fn shared_gates_remove_one_affine_map_and_respect_sequence_boundaries() {
        use crate::head::ReadGates;
        for read in [PeerRead::SHARED_GRAPH, PeerRead::CURRENT_GRAPH] {
            let width = 64;
            let count = |r| super::parameter_count(Head::PeerTriplePhase(r), width, 97);
            assert_eq!(count(PeerRead::UNIT_GRAPH) - count(read), width + 1);
        }
        for position in 0..6 {
            let prior = (position % 3 != 0).then(|| position - 1);
            assert_eq!(ReadGates::SharedPrevious.position(position, 3), prior);
            assert_eq!(
                ReadGates::SharedCurrent.position(position, 3),
                Some(position)
            );
        }
    }

    #[test]
    fn peer_trunk_and_phase_share_the_same_bounded_dictionary() {
        for search in [
            PeerSearch::Aligned,
            PeerSearch::Witnesses,
            PeerSearch::Iterated(None),
            PeerSearch::Iterated(Some(crate::slot_orbits::RoleEvidence::OBSERVED_ROLES)),
        ] {
            let read = Head::peer_phase(search).peer_read().unwrap();
            for head in [Head::PeerTriple(read), Head::PeerTriplePhase(read)] {
                assert_eq!(capacity(head, 64, 97), 97);
                assert_eq!(capacity(head, 64, 256), 256);
                assert_eq!(capacity(head, 64, 257), 64);
                assert_eq!(capacity(head, 192, 257), 128);
            }
        }
    }
}
