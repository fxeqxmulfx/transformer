//! An append-only planar upper envelope for scalar distance scores.
//! Each occupied binary bucket stores sorted upper-hull lines. The
//! intersection formula cancels the common quadratic before evaluation.
#[derive(Clone, Copy, Debug)]
pub(super) struct Line {
    pub address: f64,
    pub log_weight: f64,
    pub index: usize,
}
impl Line {
    pub fn score(self, q: f64) -> f64 {
        self.log_weight - (q - self.address).powi(2)
    }
    fn later(self, b: Self) -> Self {
        if self.index >= b.index { self } else { b }
    }
    fn winner(self, b: Self, q: f64) -> Self {
        match self.score(q).total_cmp(&b.score(q)) {
            std::cmp::Ordering::Less => b,
            std::cmp::Ordering::Greater => self,
            std::cmp::Ordering::Equal => self.later(b),
        }
    }
    fn crossing(self, right: Self) -> f64 {
        (self.address + right.address) * 0.5
            + (self.log_weight - right.log_weight) / (2. * (right.address - self.address))
    }
}
#[derive(Clone, Copy)]
struct Segment {
    line: Line,
    start: f64,
    tie: Line,
}
struct Hull(Vec<Segment>);
impl Hull {
    fn one(line: Line) -> Self {
        Self(vec![Segment {
            line,
            start: f64::NEG_INFINITY,
            tie: line,
        }])
    }
    fn merge(self, other: Self) -> Self {
        let mut lines = self.0.into_iter().map(|s| s.line).peekable();
        let mut others = other.0.into_iter().map(|s| s.line).peekable();
        let mut hull: Vec<Segment> = Vec::new();
        while lines.peek().is_some() || others.peek().is_some() {
            let line = match (lines.peek(), others.peek()) {
                (Some(a), Some(b)) if a.address == b.address => {
                    let a = lines.next().unwrap();
                    let b = others.next().unwrap();
                    a.winner(b, a.address)
                }
                (Some(a), Some(b)) if a.address < b.address => lines.next().unwrap(),
                (Some(_), Some(_)) | (None, Some(_)) => others.next().unwrap(),
                (Some(_), None) => lines.next().unwrap(),
                _ => unreachable!(),
            };
            let mut start = f64::NEG_INFINITY;
            while let Some(prev) = hull.last() {
                start = prev.line.crossing(line);
                if start >= prev.start {
                    break;
                }
                hull.pop();
            }
            let tie = match hull.last() {
                Some(prev) if prev.start == start => prev.tie.later(line),
                Some(prev) => prev.line.later(line),
                None => {
                    start = f64::NEG_INFINITY;
                    line
                }
            };
            hull.push(Segment { line, start, tie });
        }
        Self(hull)
    }
    fn best(&self, q: f64) -> Line {
        let i = self.0.partition_point(|s| s.start <= q).saturating_sub(1);
        let s = self.0[i];
        let mut best = s.line;
        if s.start == q {
            best = best.winner(s.tie, q);
        }
        // Direct distance scores on adjacent lines resolve rounded intersections.
        if i > 0 {
            best = best.winner(self.0[i - 1].line, q);
        }
        if let Some(next) = self.0.get(i + 1) {
            best = best.winner(next.line, q);
        }
        best
    }
}
#[derive(Default)]
pub(super) struct Index {
    levels: Vec<Option<Hull>>,
}
impl Index {
    pub fn write(&mut self, line: Line) {
        let mut merged = Hull::one(line);
        for layer in &mut self.levels {
            let Some(previous) = layer.take() else {
                *layer = Some(merged);
                return;
            };
            merged = previous.merge(merged);
        }
        self.levels.push(Some(merged));
    }
    pub fn best(&self, q: f64) -> Option<Line> {
        self.levels
            .iter()
            .flatten()
            .map(|h| h.best(q))
            .reduce(|a, b| a.winner(b, q))
    }
    pub fn clear(&mut self) {
        self.levels.clear();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_prefix_matches_exhaustive_distance_reads_including_latest_ties() {
        let mut lines = Vec::new();
        let mut index = Index::default();
        assert!(index.best(0.).is_none());
        for k in 0..192 {
            let a = ((k * k + 31 * k + 7) % 49) as f64 - 24.;
            // Every fourth write has a zero-height lifted line at q=0.
            let line = Line {
                address: a,
                log_weight: if k % 4 == 0 {
                    a * a
                } else {
                    -((k * 19 % 13) as f64)
                },
                index: k,
            };
            lines.push(line);
            index.write(line);
            for j in -53..=53 {
                let q = f64::from(j) / 2.;
                let scan = lines.iter().copied().reduce(|a, b| a.winner(b, q)).unwrap();
                assert_eq!(
                    index.best(q).unwrap().index,
                    scan.index,
                    "prefix {k}, q {q}"
                );
            }
        }
        index.clear();
        assert!(index.best(-7.).is_none());
    }
}
