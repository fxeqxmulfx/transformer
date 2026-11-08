// Appended to the pinned original engine. Source kernels/AdamW are unchanged.
// PROTOCOL.md defines all objective weights, detachments and shared-E paths.
use crate::config::{Aux, Kernels, Method};

impl Engine {
    fn pair_error(&self, kernels: &Kernels, batch: usize) -> Result<()> {
        // SAFETY: forward supplied valid batch logits and labels; all buffers are
        // preallocated for at most 512 rows, and launches share the source stream.
        unsafe {
            kernels.loss.pairwise(
                rows_grid(batch),
                self.dlogits.ptr(),
                self.losses.ptr(),
                self.logits.ptr(),
                self.labels.ptr(),
                batch as u32,
            )?;
        }
        Ok(())
    }

    /// L2 derivatives for readout E and B only; u and x0 are held fixed.
    pub fn local_output(&self, kernels: &Kernels, batch: usize) -> Result<()> {
        self.pair_error(kernels, batch)?;
        self.linear_back(
            self.dlast.ptr(),
            self.dlogits.ptr(),
            self.last.ptr(),
            0,
            batch,
            97,
            W,
        )?;
        self.rms_back(self.dx1.ptr(), self.x1.ptr(), self.dlast.ptr(), batch)?;
        self.rms_back(self.dbranch.ptr(), self.branch.ptr(), self.dx1.ptr(), batch)?;
        self.transpose(self.dbranch.ptr(), batch, W)?;
        self.mm(
            self.gradients.at(E + 2 * self.arm.hidden() * W),
            self.scratch.ptr(),
            self.hidden.ptr(),
            W,
            self.arm.hidden(),
            batch,
        )
    }

    /// Differentiate the unchanged source gate/product within its local block.
    fn gate_derivative(&self, batch: usize) -> Result<()> {
        let h = self.arm.hidden();
        // SAFETY: the original forward and auxiliary RMS derivative populated
        // gv and dhidden with the source's real/imaginary plane layout.
        unsafe {
            match self.arm {
                Arm::Swiglu => self.ops.swiglu_bwd(
                    grid(batch * h),
                    self.dgv.ptr(),
                    self.gv.ptr(),
                    self.dhidden.ptr(),
                    h as u32,
                    (batch * h) as u32,
                )?,
                Arm::Diagonal => self.product.product_bwd::<0>(
                    grid(batch * h / 2),
                    self.dgv.ptr(),
                    self.gv.ptr(),
                    self.dhidden.ptr(),
                    (h / 2) as u32,
                    (batch * h / 2) as u32,
                )?,
                Arm::Product => self.product.product_bwd::<1>(
                    grid(batch * h / 2),
                    self.dgv.ptr(),
                    self.gv.ptr(),
                    self.dhidden.ptr(),
                    (h / 2) as u32,
                    (batch * h / 2) as u32,
                )?,
                Arm::Plain => self.product.product_bwd::<2>(
                    grid(batch * h / 2),
                    self.dgv.ptr(),
                    self.gv.ptr(),
                    self.dhidden.ptr(),
                    (h / 2) as u32,
                    (batch * h / 2) as u32,
                )?,
            }
        }
        Ok(())
    }

    /// L1: overwrite A's gradient; optionally add the path into input E.
    pub fn hidden_objective(
        &self,
        aux: &Aux,
        kernels: &Kernels,
        batch: usize,
        propagate: bool,
    ) -> Result<()> {
        let h = self.arm.hidden();
        // SAFETY: one warp per row; padded iterations test all 341/682 bounds.
        unsafe {
            if self.arm == Arm::Swiglu {
                kernels.norm.rms::<341>(
                    rows_grid(batch),
                    aux.norm.ptr(),
                    self.hidden.ptr(),
                    batch as u32,
                )?;
            } else {
                kernels.norm.rms::<682>(
                    rows_grid(batch),
                    aux.norm.ptr(),
                    self.hidden.ptr(),
                    batch as u32,
                )?;
            }
        }
        self.mm(
            self.logits.ptr(),
            aux.norm.ptr(),
            aux.r1t.ptr(),
            batch,
            97,
            h,
        )?;
        self.pair_error(kernels, batch)?;
        self.mm(
            aux.grad.ptr(),
            self.dlogits.ptr(),
            aux.r1.ptr(),
            batch,
            h,
            97,
        )?;
        // SAFETY: disjoint auxiliary gradient and source dhidden output buffers.
        unsafe {
            if self.arm == Arm::Swiglu {
                kernels.norm.derivative::<341>(
                    rows_grid(batch),
                    self.dhidden.ptr(),
                    self.hidden.ptr(),
                    aux.grad.ptr(),
                    batch as u32,
                )?;
            } else {
                kernels.norm.derivative::<682>(
                    rows_grid(batch),
                    self.dhidden.ptr(),
                    self.hidden.ptr(),
                    aux.grad.ptr(),
                    batch as u32,
                )?;
            }
        }
        self.gate_derivative(batch)?;
        self.transpose(self.dgv.ptr(), batch, 2 * h)?;
        self.mm(
            self.gradients.at(E),
            self.scratch.ptr(),
            self.n1.ptr(),
            2 * h,
            W,
            batch,
        )?;
        if propagate {
            // Only auxiliary-bp crosses this boundary. Existing readout/global
            // E gradients are retained and source embed_bwd adds this path.
            self.mm(
                self.dn1.ptr(),
                self.dgv.ptr(),
                self.parameters.at(E),
                batch,
                W,
                2 * h,
            )?;
            self.rms_back(self.dx0.ptr(), self.x0.ptr(), self.dn1.ptr(), batch)?;
            self.rms_back(self.dembed.ptr(), self.embed.ptr(), self.dx0.ptr(), batch)?;
            unsafe {
                self.rotate.embed_bwd(
                    grid(E),
                    self.gradients.ptr(),
                    self.dembed.ptr(),
                    self.tokens.ptr(),
                    batch as u32,
                )?;
            }
        }
        Ok(())
    }

    /// L0: classify each of the 97 known vocabulary identities once, then add
    /// its direct E derivative. There are no answer labels or encoder paths.
    pub fn vocabulary_objective(&self, aux: &Aux, kernels: &Kernels) -> Result<()> {
        // SAFETY: parameters[0..E] is a 97x128 table; identity[0..97] is 0..97.
        unsafe {
            kernels
                .norm
                .rms::<128>(rows_grid(97), aux.norm.ptr(), self.parameters.ptr(), 97)?;
        }
        self.mm(self.logits.ptr(), aux.norm.ptr(), aux.r0t.ptr(), 97, 97, W)?;
        unsafe {
            kernels.loss.pairwise(
                rows_grid(97),
                self.dlogits.ptr(),
                self.losses.ptr(),
                self.logits.ptr(),
                self.identity.ptr(),
                97,
            )?;
        }
        self.mm(aux.grad.ptr(), self.dlogits.ptr(), aux.r0.ptr(), 97, W, 97)?;
        unsafe {
            kernels.norm.derivative::<128>(
                rows_grid(97),
                self.dembed.ptr(),
                self.parameters.ptr(),
                aux.grad.ptr(),
                97,
            )?;
            self.ops.add(
                grid(E),
                self.gradients.ptr(),
                self.gradients.ptr(),
                self.dembed.ptr(),
                E as u32,
            )?;
        }
        Ok(())
    }

    /// Collect all applicable derivatives at one parameter version. The
    /// source optimizer receives their sum only after this function returns.
    pub fn pilot_gradient(
        &self,
        method: Method,
        aux: Option<&Aux>,
        kernels: &Kernels,
        batch: usize,
    ) -> Result<()> {
        if batch == 0 || batch > 512 {
            return Err("invalid local training batch".into());
        }
        match method {
            Method::Ce => self.backward(batch)?,
            Method::Pair => {
                self.pair_error(kernels, batch)?;
                self.backward(batch)?;
            }
            Method::Auxiliary => {
                let aux = aux.ok_or("missing auxiliary heads")?;
                let saved = aux
                    .saved_a
                    .as_ref()
                    .ok_or("missing accumulated A gradient")?;
                self.pair_error(kernels, batch)?;
                self.backward(batch)?;
                unsafe {
                    kernels.norm.copy(
                        grid(saved.len()),
                        saved.ptr(),
                        self.gradients.at(E),
                        saved.len() as u32,
                    )?;
                }
                self.hidden_objective(aux, kernels, batch, true)?;
                unsafe {
                    self.ops.add(
                        grid(saved.len()),
                        self.gradients.at(E),
                        self.gradients.at(E),
                        saved.ptr(),
                        saved.len() as u32,
                    )?;
                }
                self.vocabulary_objective(aux, kernels)?;
            }
            Method::Local => {
                let aux = aux.ok_or("missing auxiliary heads")?;
                self.local_output(kernels, batch)?;
                self.hidden_objective(aux, kernels, batch, false)?;
                self.vocabulary_objective(aux, kernels)?;
            }
        }
        Ok(())
    }

    pub fn pilot_step(
        &self,
        method: Method,
        aux: Option<&Aux>,
        kernels: &Kernels,
        offset: usize,
        batch: usize,
        step: usize,
    ) -> Result<()> {
        self.forward(false, offset, batch, true)?;
        self.pilot_gradient(method, aux, kernels, batch)?;
        self.update(step)
    }

    /// Explicit buffer payload, including retained source training/sampler
    /// storage. This excludes CUDA context/module/allocator overhead.
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
}
