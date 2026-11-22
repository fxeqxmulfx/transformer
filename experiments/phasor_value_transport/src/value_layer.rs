// Included in the pinned engine to share its RMS, tied projection and GEMM kernels.
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
        let rows = e.config.max_batch * e.config.max_length;
        let (d, p) = (e.config.width, e.config.vocab);
        Ok(Self {
            memory: (e.config.head == crate::head::Head::Relations)
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

    // The score-gradient buffer is free until after the embedding pullback.
    // At length <= 2 the causal displacement is identically zero.
    fn value_weights(&self, length: usize) -> &F {
        if crate::pair_memory::DISPLACEMENT != 0.0
            && length > 2
            && let Some(m) = &self.memory
        {
            &m.dscores
        } else {
            &self.weights
        }
    }

    fn project(&self, e: &Engine, batch: usize, length: usize, product: bool) -> Result<()> {
        let (rows, p, d) = (batch * length, e.config.vocab, e.config.width);
        if let Some(m) = &self.memory {
            unsafe {
                m.causal.weights(
                    Grid::new(rows as u32, 32),
                    self.weights.ptr(),
                    e.logits.ptr(),
                    e.tokens.ptr(),
                    p as u32,
                    length as u32,
                )?;
                if crate::pair_memory::DISPLACEMENT != 0.0 && length > 2 {
                    m.causal.displace(
                        Grid::new(rows as u32, 128),
                        m.dscores.ptr(),
                        self.weights.ptr(),
                        e.tokens.ptr(),
                        m.keys.ptr(),
                        p as u32,
                        crate::pair_memory::TABLES as u32,
                        crate::pair_memory::DISPLACEMENT,
                    )?;
                    if let Some(frequency) = &m.frequency {
                        frequency.center(
                            grid(batch * p),
                            m.dscores.ptr(),
                            e.tokens.ptr(),
                            p as u32,
                            length as u32,
                            (batch * p) as u32,
                            0.5,
                        )?;
                    }
                    if let Some(paths) = &m.relations {
                        paths.kernels.add::<{ crate::relation_paths::MEAN }>(
                            grid(rows),
                            m.dscores.ptr(),
                            paths.endpoints.ptr(),
                            e.tokens.ptr(),
                            e.parameters
                                .at(crate::pair_memory::gain(&e.config) + 1)
                                .ok_or("relation amplitudes outside parameter buffer")?,
                            paths.scales.ptr(),
                            p as u32,
                            rows as u32,
                        )?;
                    }
                }
            }
        } else {
            // The historical two-token transport control is a separate experiment.
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
        }
        if let Some(pipeline) = &e.pipeline {
            pipeline.multiply::<{ crate::precision::BF16 }>(
                self.hidden.ptr(),
                self.value_weights(length).ptr(),
                e.parameters.ptr(),
                rows,
                d,
                p,
            )?;
        } else {
            e.mm_backward(
                self.hidden.ptr(),
                self.value_weights(length).ptr(),
                e.parameters.ptr(),
                rows,
                d,
                p,
            )?;
        }
        if let Some(m) = &self.memory {
            unsafe {
                m.kernels.read(
                    Grid::new(rows as u32, d as u32),
                    self.hidden.ptr(),
                    e.parameters
                        .at(crate::pair_memory::start(&e.config))
                        .ok_or("value table outside parameter buffer")?,
                    m.keys.ptr(),
                    d as u32,
                    crate::pair_memory::TABLES as u32,
                )?;
            }
        }
        e.rms(self.normalized.ptr(), self.hidden.ptr(), rows)?;
        let features = if let Some(m) = &self.memory {
            let (gain_addresses, stride) = m.gain_addresses();
            unsafe {
                m.causal.combine::<{ crate::pair_memory::HIT_GAIN }, { crate::pair_memory::LEARN_BONUS }>(
                    Grid::new(rows as u32, d as u32),
                    m.merged.ptr(),
                    m.scaled.ptr(),
                    e.last.ptr(),
                    self.normalized.ptr(),
                    e.parameters
                        .at(crate::pair_memory::gain(&e.config))
                        .ok_or("value gain outside parameter buffer")?,
                    e.tokens.ptr(),
                    gain_addresses.ptr(),
                    stride,
                    crate::pair_memory::HIT_FACTOR,
                    d as u32,
                )?;
            }
            m.merged.ptr()
        } else {
            self.normalized.ptr()
        };
        e.linear(e.logits.ptr(), features, 0, rows, p, d)?;
        Ok(())
    }

    // Historical masked, two-token experiment. The single-model runner always
    // calls Engine::forward and uses its label normalizer on every position.
    pub fn forward(&self, e: &Engine, batch: usize, product: bool) -> Result<()> {
        self.project(e, batch, 2, product)?;
        unsafe {
            e.stable_loss.as_ref().unwrap().loss(
                rows_grid(2 * batch),
                e.dlogits.ptr(),
                e.losses.ptr(),
                e.predictions.ptr(),
                e.logits.ptr(),
                e.labels.ptr(),
                e.config.vocab as u32,
                (2 * batch) as u32,
                batch as f32,
            )?;
        }
        Ok(())
    }

    pub fn backward(&self, e: &Engine, batch: usize, product: bool) -> Result<()> {
        self.pull_back(e, batch, 2, product)
    }

    fn pull_back(&self, e: &Engine, batch: usize, length: usize, product: bool) -> Result<()> {
        let (rows, p, d) = (batch * length, e.config.vocab, e.config.width);
        e.mm_backward(
            self.dnormalized.ptr(),
            e.dlogits.ptr(),
            e.parameters.ptr(),
            rows,
            d,
            p,
        )?;
        let value_feature = if let Some(m) = &self.memory {
            let (gain_addresses, stride) = m.gain_addresses();
            let gain = crate::pair_memory::gain(&e.config);
            unsafe {
                m.causal.gain_back::<{ crate::pair_memory::HIT_GAIN }, { crate::pair_memory::LEARN_BONUS }>(
                    Grid::new(rows as u32, 32),
                    m.partial.ptr(),
                    self.dnormalized.ptr(),
                    self.normalized.ptr(),
                    e.parameters
                        .at(gain)
                        .ok_or("value gain outside parameter buffer")?,
                    e.tokens.ptr(),
                    gain_addresses.ptr(),
                    stride,
                    crate::pair_memory::HIT_FACTOR,
                    d as u32,
                    rows as u32,
                )?;
                m.causal.gain_sum(
                    Grid::new(1 + u32::from(crate::pair_memory::LEARN_BONUS), 32),
                    e.gradients
                        .at(gain)
                        .ok_or("value gain outside gradient buffer")?,
                    m.partial.ptr(),
                    rows as u32,
                )?;
            }
            m.scaled.ptr()
        } else {
            self.normalized.ptr()
        };
        e.transpose(e.dlogits.ptr(), rows, p)?;
        e.mm_backward(
            self.read_gradient.ptr(),
            e.scratch.ptr(),
            value_feature,
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
        e.transpose(self.value_weights(length).ptr(), rows, p)?;
        e.mm_backward(
            self.value_gradient.ptr(),
            e.scratch.ptr(),
            self.dhidden.ptr(),
            p,
            d,
            rows,
        )?;
        if let Some(m) = &self.memory {
            unsafe {
                if let Some(paths) = &m.relations {
                    paths.kernels.adjoint::<{ crate::relation_paths::MEAN }>(
                        Grid::new(p as u32, 32),
                        e.gradients
                            .at(crate::pair_memory::gain(&e.config) + 1)
                            .ok_or("relation amplitudes outside gradient buffer")?,
                        self.dweights.ptr(),
                        paths.endpoints.ptr(),
                        e.tokens.ptr(),
                        paths.scales.ptr(),
                        p as u32,
                        if length > 2 { rows as u32 } else { 0 },
                    )?;
                }
                m.kernels.backward(
                    Grid::new((p * crate::pair_memory::TABLES) as u32, d as u32),
                    e.gradients
                        .at(crate::pair_memory::start(&e.config))
                        .ok_or("value table outside gradient buffer")?,
                    self.dhidden.ptr(),
                    m.offsets.ptr(),
                    m.owners.ptr(),
                    d as u32,
                )?;
                m.causal.pullback(
                    Grid::new(rows as u32, 32),
                    m.dscores.ptr(),
                    self.weights.ptr(),
                    self.dweights.ptr(),
                    e.tokens.ptr(),
                    p as u32,
                    length as u32,
                )?;
                m.causal.add_sources(
                    Grid::new(rows as u32, 128),
                    e.dlogits.ptr(),
                    m.dscores.ptr(),
                    p as u32,
                    length as u32,
                )?;
            }
        } else {
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
        }
        e.backward_original(batch, length)?;
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
}

impl Engine {
    fn enable_output_head(&mut self) -> Result<()> {
        use crate::head::Head;
        match self.config.head {
            Head::Original => {}
            Head::Stream | Head::CompactStream | Head::PhasePrefix | Head::InputPhase | Head::AffinePhase | Head::BilinearPhase | Head::ReflectionPhase | Head::SingleReflection | Head::ContextRotation | Head::CrossedContext | Head::AssociativeReflection | Head::PeriodicReflection => {
                if self.mhc.is_none() {
                    return Err("the learned readout requires the four Phasor streams".into());
                }
                self.stream_read = Some(crate::learned_read::kernels::Kernels::load(&self.cuda)?);
                if self.config.head.context_stages() > 0 {
                    self.context_rotation = Some(context_rotation_layer::ContextRotation::new(self)?);
                } else if self.config.head.is_phase() {
                    self.hidden_layer = Some(hidden_layer::HiddenLayer::new(self)?);
                }
            }
            Head::Relations => self.shared_layer = Some(ValueLayer::new(self)?),
            Head::InputProduct =>
                self.input_product = Some(product_layer::ProductLayer::new(self)?),
            Head::Transition | Head::Recurrent | Head::Bounded =>
                self.hidden_layer = Some(hidden_layer::HiddenLayer::new(self)?),
        }
        Ok(())
    }
    fn project_output(&mut self, tokens: &[u32], batch: usize, length: usize) -> Result<()> {
        let rows = batch * length;
        self.transform_context(rows, length)?;
        self.classify_features(rows)?;
        if let Some(layer) = &mut self.shared_layer {
            layer.memory.as_mut().unwrap().prepare(tokens, length)?;
            self.shared_layer
                .as_ref()
                .unwrap()
                .project(self, batch, length, true)?;
        }
        Ok(())
    }
    fn output_backward(&self, rows: usize, length: usize) -> Result<()> {
        self.linear_back(
            self.dlast.ptr(), self.dlogits.ptr(), self.encoded_features().ptr(), 0,
            rows, self.config.vocab, self.config.width,
        )?;
        if let Some(memory) = &self.hidden_layer {
            memory.backward(self, rows, length)?;
        } else if let Some(rotation) = &self.context_rotation {
            rotation.backward(self, rows, length)?;
        }
        Ok(())
    }
    pub fn backward(&mut self, batch: usize, length: usize) -> Result<()> {
        if self.uses_embedding_backbone() && self.supervised_shape != Some((batch, length)) {
            return Err("supervised backward needs the latest labeled forward with the same shape".into());
        }
        if let Some(layer) = &mut self.shared_layer {
            if batch == 0
                || batch > self.config.max_batch
                || length == 0
                || length > self.config.max_length
            {
                return Err("batch or length outside allocated bounds".into());
            }
            layer
                .memory
                .as_mut()
                .unwrap()
                .prepare_transpose(batch, length)?;
            return self
                .shared_layer
                .as_ref()
                .unwrap()
                .pull_back(self, batch, length, true);
        }
        self.backward_original(batch, length)
    }
}
