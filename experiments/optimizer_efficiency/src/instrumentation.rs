// Buffer accounting adapted from forward_training/instrumentation.rs at
// c9995ba5. The source CE, backward and AdamW functions are unchanged.
impl Engine {
    pub fn allocated_bytes(&self) -> usize {
        let floats = [
            &self.parameters,
            &self.gradients,
            &self.first,
            &self.second,
            &self.embed,
            &self.x0,
            &self.n1,
            &self.gv,
            &self.hidden,
            &self.branch,
            &self.branch_norm,
            &self.x1,
            &self.last,
            &self.logits,
            &self.dlogits,
            &self.dlast,
            &self.dx1,
            &self.dbranch,
            &self.dhidden,
            &self.dgv,
            &self.dn1,
            &self.dx0,
            &self.dembed,
            &self.losses,
            &self.scratch,
        ]
        .iter()
        .map(|b| b.len())
        .sum::<usize>();
        let words = [
            &self.pairs,
            &self.train,
            &self.validation,
            &self.order,
            &self.identity,
            &self.tokens,
            &self.labels,
            &self.hits,
        ]
        .iter()
        .map(|b| b.len())
        .sum::<usize>();
        4 * (floats + words)
    }
    pub fn moment_bytes(&self) -> usize {
        4 * (self.first.len() + self.second.len())
    }
}
