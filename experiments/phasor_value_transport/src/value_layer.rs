// Included in the pinned engine module to reuse its affine, RMS, CE and adjoint kernels.
pub struct ValueLayer {
    memory: Option<crate::pair_memory::PairMemory>,
    addresses: crate::addresses::kernels::Kernels,
    weights: F,
    hidden: F,
    normalized: F,
    dnormalized: F,
    dhidden: F,
    dweights: F,
    read_gradient: F,
    value_gradient: F,
}

impl ValueLayer {
    pub fn new(e: &Engine) -> Result<Self> {
        let rows = 2 * e.config.max_batch;
        let (d, p) = (e.config.width, e.config.vocab);
        Ok(Self {
            memory: crate::pair_memory::active(&e.config)
                .then(|| crate::pair_memory::PairMemory::new(&e.cuda, &e.config))
                .transpose()?,
            addresses: crate::addresses::kernels::Kernels::load(&e.cuda)?,
            weights: e.cuda.alloc(rows * p)?,
            hidden: e.cuda.alloc(rows * d)?,
            normalized: e.cuda.alloc(rows * d)?,
            dnormalized: e.cuda.alloc(rows * d)?,
            dhidden: e.cuda.alloc(rows * d)?,
            dweights: e.cuda.alloc(rows * p)?,
            read_gradient: e.cuda.alloc(p * d)?,
            value_gradient: e.cuda.alloc(p * d)?,
        })
    }

    // Each active pair owns two contiguous rows. The first token's loss is
    // masked in the original dataset. All kernel calls share the engine stream.
    fn project(&self, e: &Engine, batch: usize, product: bool) -> Result<()> {
        let (rows, p, d) = (2 * batch, e.config.vocab, e.config.width);
        unsafe {
            self.addresses.forward(
                grid(batch),
                self.weights.ptr(),
                e.logits.ptr(),
                e.tokens.ptr(),
                p as u32,
                u32::from(product),
                batch as u32,
            )?;
        }
        // W is already indexed by the translated value address, so Z = W E.
        if let Some(pipeline) = &e.pipeline {
            pipeline.multiply::<{ crate::precision::BF16 }>(
                self.hidden.ptr(),
                self.weights.ptr(),
                e.parameters.ptr(),
                rows,
                d,
                p,
            )?;
        } else {
            e.mm_backward(
                self.hidden.ptr(),
                self.weights.ptr(),
                e.parameters.ptr(),
                rows,
                d,
                p,
            )?;
        }
        let mut output = e.logits.ptr();
        if let Some(m) = &self.memory {
            unsafe {
                m.kernels.read(
                    Grid::new(batch as u32, d as u32),
                    self.hidden.ptr(),
                    e.parameters.at(crate::pair_memory::start(&e.config)),
                    m.keys.ptr(),
                    d as u32,
                    crate::pair_memory::TABLES as u32,
                )?;
            }
            output = m.output.ptr();
        }
        e.rms(self.normalized.ptr(), self.hidden.ptr(), rows)?;
        e.linear(output, self.normalized.ptr(), 0, rows, p, d)?;
        if let Some(m) = &self.memory {
            unsafe {
                m.kernels.logits::<true, false>(
                    Grid::new(batch as u32, 128),
                    e.logits.ptr(),
                    output,
                    p as u32,
                )?;
            }
        }
        Ok(())
    }

    pub fn forward(&self, e: &Engine, batch: usize, product: bool) -> Result<()> {
        self.project(e, batch, product)?;
        let (rows, p) = (2 * batch, e.config.vocab);
        unsafe {
            e.stable_loss.as_ref().unwrap().loss(
                rows_grid(rows),
                e.dlogits.ptr(),
                e.losses.ptr(),
                e.predictions.ptr(),
                e.logits.ptr(),
                e.labels.ptr(),
                p as u32,
                rows as u32,
                batch as f32,
            )?;
        }
        Ok(())
    }

    // The shared table has three gradient paths: ordinary first-token Phasor,
    // the outer tied readout, and the transported embedding values.
    pub fn backward(&self, e: &Engine, batch: usize, product: bool) -> Result<()> {
        let (rows, p, d) = (2 * batch, e.config.vocab, e.config.width);
        if let Some(m) = &self.memory {
            unsafe {
                m.kernels.logits::<false, false>(
                    Grid::new(batch as u32, 128),
                    e.dlogits.ptr(),
                    m.output.ptr(),
                    p as u32,
                )?;
            }
        }
        e.mm_backward(
            self.dnormalized.ptr(),
            e.dlogits.ptr(),
            e.parameters.ptr(),
            rows,
            d,
            p,
        )?;
        e.transpose(e.dlogits.ptr(), rows, p)?;
        e.mm_backward(
            self.read_gradient.ptr(),
            e.scratch.ptr(),
            self.normalized.ptr(),
            p,
            d,
            rows,
        )?;
        e.rms_back(
            self.dhidden.ptr(),
            self.hidden.ptr(),
            self.dnormalized.ptr(),
            rows,
        )?;
        e.transpose(e.parameters.ptr(), p, d)?;
        e.mm_backward(
            self.dweights.ptr(),
            self.dhidden.ptr(),
            e.scratch.ptr(),
            rows,
            p,
            d,
        )?;
        e.transpose(self.weights.ptr(), rows, p)?;
        e.mm_backward(
            self.value_gradient.ptr(),
            e.scratch.ptr(),
            self.dhidden.ptr(),
            p,
            d,
            rows,
        )?;
        unsafe {
            self.addresses.backward(
                grid(batch),
                e.dlogits.ptr(),
                self.weights.ptr(),
                self.dweights.ptr(),
                e.tokens.ptr(),
                p as u32,
                u32::from(product),
                batch as u32,
            )?;
        }
        if let Some(m) = &self.memory {
            unsafe {
                m.kernels.logits::<false, true>(
                    Grid::new(batch as u32, 128),
                    e.dlogits.ptr(),
                    m.output.ptr(),
                    p as u32,
                )?;
            }
            self.table_backward::<false>(e)?;
        }
        e.backward_original(batch, 2)?;
        for extra in [&self.read_gradient, &self.value_gradient] {
            unsafe {
                e.ops.add(
                    grid(p * d),
                    e.gradients.ptr(),
                    e.gradients.ptr(),
                    extra.ptr(),
                    (p * d) as u32,
                )?;
            }
        }
        Ok(())
    }

    fn table_backward<const CLEAR: bool>(&self, e: &Engine) -> Result<()> {
        let m = self.memory.as_ref().unwrap();
        let (p, d) = (e.config.vocab, e.config.width);
        unsafe {
            m.kernels.backward::<CLEAR>(
                Grid::new((p * crate::pair_memory::TABLES) as u32, d as u32),
                e.gradients.at(crate::pair_memory::start(&e.config)),
                self.dhidden.ptr(),
                m.offsets.ptr(),
                m.owners.ptr(),
                d as u32,
            )?;
        }
        Ok(())
    }
}

impl Engine {
    fn enable_pair_values(&mut self) -> Result<()> {
        if crate::pair_memory::active(&self.config) {
            self.pair_layer = Some(ValueLayer::new(self)?);
        }
        Ok(())
    }

    fn project_pair_values(&mut self, tokens: &[u32], batch: usize, length: usize) -> Result<()> {
        if length == 2
            && let Some(layer) = &mut self.pair_layer
        {
            layer.memory.as_mut().unwrap().prepare(tokens)?;
            self.pair_layer
                .as_ref()
                .unwrap()
                .project(self, batch, true)?;
        }
        Ok(())
    }

    pub fn backward(&self, batch: usize, length: usize) -> Result<()> {
        if let Some(layer) = &self.pair_layer {
            if batch == 0
                || batch > self.config.max_batch
                || length == 0
                || length > self.config.max_length
            {
                return Err("batch or length outside allocated bounds".into());
            }
            if length == 2 {
                return layer.backward(self, batch, true);
            }
            layer.table_backward::<true>(self)?;
        }
        self.backward_original(batch, length)
    }
}
