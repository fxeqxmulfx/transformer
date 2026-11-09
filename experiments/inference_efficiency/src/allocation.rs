// Engine::new from trainer 14cff3b5, with module loading moved before allocation.
// Every buffer shape and validation is retained for honest memory accounting.
impl Engine {
    pub fn with_modules(
        arm: Arm,
        p: &[f32],
        data: &Data,
        plan: &Plan,
        m: crate::device::Modules,
    ) -> Result<Self> {
        if !model::valid(arm, p)
            || data.train.len() != 4704 * 3
            || data.validation.len() != 4705 * 3
            || data.train.iter().chain(&data.validation).any(|&x| x >= 97)
            || plan.order.is_empty()
            || plan.order.iter().any(|&x| x >= 4704)
        {
            return Err("invalid model or dataset".into());
        }
        let cuda = Cuda::new()?;
        let h = arm.hidden();
        let pairs: Vec<u32> = model::pairs().into_iter().flatten().collect();
        Ok(Self {
            arm,
            parameters: cuda.upload(p)?,
            gradients: cuda.alloc(p.len())?,
            first: cuda.alloc(p.len())?,
            second: cuda.alloc(p.len())?,
            ops: m.ops,
            gemm: m.gemm,
            product: m.product,
            rotate: m.rotate,
            pairs: cuda.upload(&pairs)?,
            train: cuda.upload(&data.train)?,
            validation: cuda.upload(&data.validation)?,
            order: cuda.upload(&plan.order)?,
            identity: cuda.upload(&(0..4705).collect::<Vec<u32>>())?,
            tokens: cuda.alloc(1024)?,
            labels: cuda.alloc(512)?,
            embed: cuda.alloc(512 * W)?,
            x0: cuda.alloc(512 * W)?,
            n1: cuda.alloc(512 * W)?,
            gv: cuda.alloc(512 * 2 * h)?,
            hidden: cuda.alloc(512 * h)?,
            branch: cuda.alloc(512 * W)?,
            branch_norm: cuda.alloc(512 * W)?,
            x1: cuda.alloc(512 * W)?,
            last: cuda.alloc(512 * W)?,
            logits: cuda.alloc(512 * 97)?,
            dlogits: cuda.alloc(512 * 97)?,
            dlast: cuda.alloc(512 * W)?,
            dx1: cuda.alloc(512 * W)?,
            dbranch: cuda.alloc(512 * W)?,
            dhidden: cuda.alloc(512 * h)?,
            dgv: cuda.alloc(512 * 2 * h)?,
            dn1: cuda.alloc(512 * W)?,
            dx0: cuda.alloc(512 * W)?,
            dembed: cuda.alloc(512 * W)?,
            losses: cuda.alloc(512)?,
            hits: cuda.alloc(512)?,
            scratch: cuda.alloc(512 * 2 * h)?,
        })
    }
    pub fn serving_allocated_bytes(&self) -> usize {
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
}
