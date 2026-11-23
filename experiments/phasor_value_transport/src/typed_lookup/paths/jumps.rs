//! Materialize powers of the transition on one observed causal prefix.
//! A start state enters its first edge. A cursor state carries the complete
//! directed write, including the time needed by ordered reads.
use super::{Bindings, Edge, Store, Writes};

pub(super) struct Jumps {
    edges: Vec<Edge>,
    by_time: Vec<usize>,
    levels: Vec<Vec<usize>>,
}

impl Jumps {
    // Each observed triple writes once in each direction at its final position.
    fn stamp(edge: Edge, relation: usize) -> usize {
        2 * edge.written_at + relation % 2
    }

    pub(super) fn build(store: &Store, budget: usize) -> Self {
        assert!(budget > 0);
        let mut by_time = Vec::new();
        let mut edges = Vec::new();
        let mut relations = Vec::new();
        for bindings in store.outgoing.values().flatten() {
            let records: &[Edge] = match &bindings.writes {
                Writes::Latest(last) => std::slice::from_ref(last),
                Writes::Ordered(history) => history,
            };
            for &edge in records {
                let stamp = Self::stamp(edge, bindings.relation);
                if stamp >= by_time.len() {
                    by_time.resize(stamp + 1, usize::MAX);
                }
                debug_assert_eq!(by_time[stamp], usize::MAX);
                by_time[stamp] = edges.len();
                edges.push(edge);
                relations.push(bindings.relation);
            }
        }
        let mut one = Vec::with_capacity(2 * edges.len());
        for (i, (&edge, &relation)) in edges.iter().zip(&relations).enumerate() {
            one.push(
                store
                    .successor(relation, edge)
                    .map_or(i, |e| by_time[Self::stamp(e, relation)]),
            );
        }
        // The second half of level zero represents a query before its first read.
        one.extend(0..edges.len());
        let mut levels = vec![one];
        for j in 0..budget.ilog2() as usize {
            let prev = &levels[j];
            levels.push(prev.iter().map(|&i| prev[i]).collect());
        }
        Self {
            edges,
            by_time,
            levels,
        }
    }

    pub(super) fn endpoints(
        &self,
        first: &Bindings,
        budgets: &[usize],
        mut sample: impl FnMut(usize, Edge),
    ) {
        let edge = *first.writes.read(None).expect("empty relation history");
        let query = self.edges.len() + self.by_time[Self::stamp(edge, first.relation)];
        for (i, &budget) in budgets.iter().enumerate() {
            let mut cursor = query;
            let mut remaining = budget;
            while remaining > 0 {
                cursor = self.levels[remaining.trailing_zeros() as usize][cursor];
                remaining &= remaining - 1;
            }
            sample(i, self.edges[cursor]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_prefix_keeps_all_directed_witnesses_and_their_order() {
        let budgets = [1, 2, 3, 3, 8, 11, 39, 40, 41, 80];
        for ordered in [false, true] {
            let mut store = Store::new(ordered);
            for i in 0..97 {
                let key = (i % 17 != 0).then_some((i * i + i / 7) % 9);
                store.observe(key, i);
                let mut expected = vec![Vec::new(); budgets.len()];
                let mut actual = expected.clone();
                store.visit_budgets(&budgets, super::super::Walk::Sequential, |j, w| {
                    expected[j].push(w)
                });
                store.visit_budgets(&budgets, super::super::Walk::JumpTables, |j, w| {
                    actual[j].push(w)
                });
                assert_eq!(actual, expected);
            }
        }
    }
}
