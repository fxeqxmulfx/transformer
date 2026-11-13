//! CM34 actual-range compaction. State ids use first active-pair occurrence,
//! so any invertible basis change preserves the state partition and ordering.
//! Unmatched states and missing axes remain distinct recorded outcomes.
use crate::*;
use std::collections::BTreeMap;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Encoding {
    pub code: discovery::Code,
    pub keys: Vec<Vec<u16>>,
    pub cell_blocks: Vec<Option<usize>>,
    /// Original observed class names in fixed bank order, possibly aliased.
    pub classes: Vec<Vec<usize>>,
    pub fallback_block: usize,
    pub unknown_axis_cells: usize,
    pub unmatched_active_cells: usize,
    pub vector_additions: usize,
}
impl Encoding {
    pub fn new(n: usize, vocabulary: usize, code: &discovery::Code) -> Result<Self> {
        limits::dimensions(n, vocabulary)?;
        code.validate(n, vocabulary)?;
        let (mut keys, mut indices) = (vec![], BTreeMap::new());
        let mut cell_blocks = vec![None; n * n];
        for &column in &code.names.columns {
            for &row in &code.names.rows {
                let key = code
                    .query(row, column)
                    .ok_or("active coordinate lacks a vector")?;
                let index = *indices.entry(key.clone()).or_insert_with(|| {
                    let index = keys.len();
                    keys.push(key);
                    index
                });
                cell_blocks[row + n * column] = Some(index);
            }
        }
        let mut classes = vec![vec![]; keys.len()];
        for (&label, key) in code.names.classes.iter().zip(&code.classes) {
            let index = *indices
                .get(key)
                .ok_or("observed class code absent from active query range")?;
            classes[index].push(label);
        }
        let fallback_block = *indices
            .get(&code.classes[0])
            .ok_or("fallback class is unreachable")?;
        let unknown_axis_cells = cell_blocks.iter().filter(|b| b.is_none()).count();
        let unmatched_active_cells = cell_blocks
            .iter()
            .flatten()
            .filter(|&&b| classes[b].is_empty())
            .count();
        Ok(Self {
            code: code.clone(),
            keys,
            cell_blocks,
            classes,
            fallback_block,
            unknown_axis_cells,
            unmatched_active_cells,
            vector_additions: code.names.rows.len() * code.names.columns.len() * code.dimension,
        })
    }
    pub fn validate(&self, n: usize, vocabulary: usize) -> Result<()> {
        if Self::new(n, vocabulary, &self.code)? != *self {
            return Err("actual vector range, class bank, fallback or masks changed".into());
        }
        Ok(())
    }
    pub fn matched(&self, cell: usize) -> bool {
        self.cell_blocks
            .get(cell)
            .copied()
            .flatten()
            .is_some_and(|b| !self.classes[b].is_empty())
    }
    pub fn default_label(&self, block: usize) -> usize {
        self.classes[block]
            .first()
            .copied()
            .unwrap_or(self.code.names.classes[0])
    }
}
