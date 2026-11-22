//! A tied dense spectrum matrix and two affine controls from the strict prefix.
use super::tied_projection::TiedProjection;
use super::*;
use crate::fourier::kernels::Kernels;

pub(crate) struct FourierProbe {
    pub coordinates: Vec<f32>,
    pub routes: Vec<f32>,
    pub phases: Vec<f32>,
}

pub(super) struct FourierLayer {
    kernels: Kernels,
    m: usize,
    weights: usize,
    control_weights: usize,
    bias: usize,
    tied: TiedProjection,
    spectrum: F,
    dspectrum: F,
    route: F,
    phase: F,
    controls: F,
    dcontrols: F,
    transformed: F,
    dtransformed: F,
    dmixed: F,
    dangle: F,
    droute: F,
    output: F,
    dinput: F,
    ones: F,
    shape: std::cell::Cell<Option<(usize, usize)>>,
}

impl FourierLayer {
    pub(super) fn new(e: &Engine) -> Result<Self> {
        let d = e.config.width;
        let m = crate::fourier::channels(d);
        let n = 2 * m + 1;
        let r = e.config.max_batch * e.config.max_length;
        let weights = crate::architecture::memory_start(&e.config);
        let control_weights = weights + 2 * m * d;
        Ok(Self {
            m,
            weights,
            control_weights,
            bias: control_weights + n * d,
            kernels: Kernels::load(&e.cuda)?,
            tied: TiedProjection::new(e, weights, 2 * m)?,
            spectrum: e.cuda.alloc(r * 2 * m)?,
            dspectrum: e.cuda.alloc(r * 2 * m)?,
            route: e.cuda.alloc(r * (m + 1))?,
            phase: e.cuda.alloc(r * 2 * m)?,
            controls: e.cuda.alloc(r * n)?,
            dcontrols: e.cuda.alloc(r * n)?,
            transformed: e.cuda.alloc(r * 2 * m)?,
            dtransformed: e.cuda.alloc(r * 2 * m)?,
            dmixed: e.cuda.alloc(r * 2 * m)?,
            dangle: e.cuda.alloc(r * m)?,
            droute: e.cuda.alloc(r * (m + 1))?,
            output: e.cuda.alloc(r * d)?,
            dinput: e.cuda.alloc(r * d)?,
            ones: e.cuda.upload(&vec![1.; r])?,
            shape: std::cell::Cell::new(None),
        })
    }
    pub(super) fn features(&self) -> &F {
        &self.output
    }
    pub(super) fn forward(&self, e: &Engine, rows: usize, length: usize) -> Result<()> {
        self.shape.set(None);
        let (m, d) = (self.m, e.config.width);
        e.linear(
            self.spectrum.ptr(),
            e.embed_norm.ptr(),
            self.weights,
            rows,
            2 * m,
            d,
        )?;
        e.linear(
            self.controls.ptr(),
            e.last.ptr(),
            self.control_weights,
            rows,
            2 * m + 1,
            d,
        )?;
        let bias = e
            .parameters
            .at(self.bias)
            .ok_or("missing Fourier control biases")?;
        unsafe {
            self.kernels.prepare(
                Grid::new(rows as u32, 32),
                self.route.ptr(),
                self.phase.ptr(),
                self.controls.ptr(),
                bias,
                m as u32,
                length as u32,
            )?;
            self.kernels.mix::<false>(
                Grid::new(rows as u32, m as u32),
                self.transformed.ptr(),
                self.spectrum.ptr(),
                self.route.ptr(),
                self.phase.ptr(),
            )?;
        }
        self.tied
            .forward(e, self.output.ptr(), self.transformed.ptr(), rows)?;
        self.shape.set(Some((rows, length)));
        Ok(())
    }
    pub(super) fn backward(&self, e: &Engine, rows: usize, length: usize) -> Result<()> {
        if self.shape.get() != Some((rows, length)) {
            return Err("Fourier backward needs a matching forward".into());
        }
        let (m, d) = (self.m, e.config.width);
        let at = Grid::new(rows as u32, m as u32);
        self.tied
            .backward(e, self.dtransformed.ptr(), self.transformed.ptr(), rows)?;
        unsafe {
            self.kernels.unrotate(
                at,
                self.dmixed.ptr(),
                self.dangle.ptr(),
                self.dtransformed.ptr(),
                self.transformed.ptr(),
                self.phase.ptr(),
            )?;
            self.kernels.mix::<true>(
                at,
                self.dspectrum.ptr(),
                self.dmixed.ptr(),
                self.route.ptr(),
                self.phase.ptr(),
            )?;
            self.kernels.route_back(
                Grid::new(rows as u32, (m + 1) as u32),
                self.droute.ptr(),
                self.dmixed.ptr(),
                self.spectrum.ptr(),
                m as u32,
            )?;
            self.kernels.controls_back(
                Grid::new(rows as u32, 32),
                self.dcontrols.ptr(),
                self.droute.ptr(),
                self.route.ptr(),
                self.dangle.ptr(),
                m as u32,
                length as u32,
            )?;
        }
        e.linear_back(
            self.dinput.ptr(),
            self.dspectrum.ptr(),
            e.embed_norm.ptr(),
            self.weights,
            rows,
            2 * m,
            d,
        )?;
        self.tied.accumulate(e)?;
        e.linear_back(
            e.dlast.ptr(),
            self.dcontrols.ptr(),
            e.last.ptr(),
            self.control_weights,
            rows,
            2 * m + 1,
            d,
        )?;
        let dbias = e
            .gradients
            .at(self.bias)
            .ok_or("missing Fourier bias gradients")?;
        e.mm_backward(
            dbias,
            self.ones.ptr(),
            self.dcontrols.ptr(),
            1,
            2 * m + 1,
            rows,
        )
    }
    pub(super) fn add_input_gradient(&self, e: &Engine, target: P, rows: usize) -> Result<()> {
        let n = rows * e.config.width;
        unsafe {
            e.ops
                .add(grid(n), target, target, self.dinput.ptr(), n as u32)?;
        }
        Ok(())
    }
}

impl Engine {
    /// Read only the latest dense activations for post-training diagnostics.
    pub(crate) fn fourier_probe(&self) -> Result<FourierProbe> {
        let Some(context_layer::ContextLayer::Fourier(x)) = &self.context_layer else {
            return Err("Fourier probe requires the Fourier read".into());
        };
        let (rows, _) = x.shape.get().ok_or("Fourier probe requires a forward")?;
        let take = |b: &F, n: usize| -> Result<Vec<f32>> {
            let mut v = b.read()?;
            v.truncate(n);
            Ok(v)
        };
        Ok(FourierProbe {
            coordinates: take(&x.spectrum, rows * 2 * x.m)?,
            routes: take(&x.route, rows * (x.m + 1))?,
            phases: take(&x.phase, rows * 2 * x.m)?,
        })
    }
}
