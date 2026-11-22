//! Store the rows actually addressed by this backward tape. Contributions to
//! each row retain the order in which the reference adjoint adds them.
use crate::slot_memory::{Layout, Row};

pub struct Adjoint {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
    pub gates: Vec<f32>,
}

impl Adjoint {
    pub fn new(l: &Layout, tape: &[Row]) -> Self {
        let mut indices = Vec::new();
        for row in tape {
            if let Some(cell) = row.cell {
                indices.push((l.slots + cell) as u32);
            }
            if let Some(slot) = row.constant {
                indices.push(slot as u32);
            }
            if let Some(slot) = row.transport {
                indices.push(slot as u32);
            }
        }
        indices.sort_unstable();
        indices.dedup();
        Self {
            values: vec![0.; indices.len() * l.d],
            indices,
            gates: vec![0.; l.end() - l.gates],
        }
    }

    pub fn row(&mut self, address: usize, d: usize) -> &mut [f32] {
        let position = self
            .indices
            .binary_search(&(address as u32))
            .expect("address must occur in its own tape");
        &mut self.values[position * d..][..d]
    }

    pub fn dense(&self, l: &Layout) -> Vec<f32> {
        let mut full = vec![0.; l.end()];
        for (&address, gradient) in self.indices.iter().zip(self.values.chunks_exact(l.d)) {
            full[address as usize * l.d..][..l.d].copy_from_slice(gradient);
        }
        full[l.gates..].copy_from_slice(&self.gates);
        full
    }
}
