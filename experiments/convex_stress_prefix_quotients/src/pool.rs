//! Input-only nonempty prefix closure (CM58); every parent precedes its child.
use crate::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct Node {
    pub parent: Option<usize>,
    pub symbol: usize,
    pub depth: usize,
}
#[derive(Clone, Debug)]
pub struct Pool {
    pub nodes: Vec<Node>,
    pub row_nodes: Vec<usize>,
}
impl Pool {
    pub fn new(rows: &[Observation]) -> Result<Self> {
        let mut edges = BTreeMap::new();
        let mut nodes: Vec<Node> = vec![];
        let mut row_nodes = Vec::with_capacity(rows.len());
        for r in rows {
            let word = prior::prefix::payload(&r.tokens).ok_or("supplied alphabet mismatch")?;
            let mut parent = None;
            for &symbol in word {
                let key = (parent, symbol);
                let index = if let Some(&i) = edges.get(&key) {
                    i
                } else {
                    let i = nodes.len();
                    let depth = parent.map_or(1, |p: usize| nodes[p].depth + 1);
                    nodes.push(Node {
                        parent,
                        symbol,
                        depth,
                    });
                    edges.insert(key, i);
                    i
                };
                parent = Some(index);
            }
            row_nodes.push(parent.ok_or("empty payload")?);
        }
        Ok(Self { nodes, row_nodes })
    }
    pub fn word(&self, mut index: usize) -> Vec<usize> {
        let mut out = vec![];
        loop {
            let n = &self.nodes[index];
            out.push(n.symbol);
            if let Some(p) = n.parent {
                index = p;
            } else {
                break;
            }
        }
        out.reverse();
        out
    }
    pub fn signature(&self, f: &Formula) -> Result<Vec<u64>> {
        let mut out = vec![0; self.nodes.len().div_ceil(64)];
        for i in 0..self.nodes.len() {
            if *f
                .evaluate(&self.word(i))?
                .last()
                .ok_or("empty truth vector")?
            {
                out[i / 64] |= 1 << (i % 64);
            }
        }
        Ok(out)
    }
    pub fn counts(&self, signature: &[u64]) -> Vec<i16> {
        let mut out = vec![0; self.nodes.len()];
        for (i, n) in self.nodes.iter().enumerate() {
            out[i] = n.parent.map_or(0, |p| out[p]) + i16::from(bit(signature, i));
        }
        out
    }
}
pub fn bit(bits: &[u64], i: usize) -> bool {
    bits[i / 64] & (1 << (i % 64)) != 0
}
pub fn mask(bits: &mut [u64], n: usize) {
    if !n.is_multiple_of(64) {
        *bits.last_mut().unwrap() &= (1 << (n % 64)) - 1;
    }
}
