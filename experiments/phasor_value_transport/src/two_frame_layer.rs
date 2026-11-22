//! Two learned one-parameter rotations around a common affine projection.
use super::*;
use crate::two_frame::kernels::Kernels;

struct Frame {
    position: F,
    dposition: F,
    angles: F,
    dangles: F,
    rotated: F,
    dx: F,
    frequency: usize,
}

impl Frame {
    fn new(e: &Engine, rows: usize, frequency: usize) -> Result<Self> {
        let d = e.config.width;
        Ok(Self {
            position: e.cuda.alloc(rows)?,
            dposition: e.cuda.alloc(rows)?,
            angles: e.cuda.alloc(rows * (d / 2))?,
            dangles: e.cuda.alloc(rows * (d / 2))?,
            rotated: e.cuda.alloc(rows * d)?,
            dx: e.cuda.alloc(rows * d)?,
            frequency,
        })
    }

    fn angles(&self, e: &Engine, rows: usize) -> Result<()> {
        e.linear(
            self.angles.ptr(),
            self.position.ptr(),
            self.frequency,
            rows,
            e.config.width / 2,
            1,
        )
    }
    fn position_back(&self, e: &Engine, rows: usize) -> Result<()> {
        e.linear_back(
            self.dposition.ptr(),
            self.dangles.ptr(),
            self.position.ptr(),
            self.frequency,
            rows,
            e.config.width / 2,
            1,
        )
    }
}

pub(super) struct TwoFrame {
    kernels: Kernels,
    input: Frame,
    output: Frame,
    ctrl: F,
    dctrl: F,
    gain: F,
    bias: F,
    dbias: F,
    middle: F,
    dmiddle_input: F,
    final_feature: F,
    dcontext: F,
    controls: usize,
    bias_weights: usize,
    matrix: usize,
    shape: std::cell::Cell<Option<(usize, usize)>>,
}

impl TwoFrame {
    pub(super) fn new(e: &Engine) -> Result<Self> {
        let cfg = &e.config;
        let d = cfg.width;
        let rows = cfg.max_batch * cfg.max_length;
        let controls = crate::architecture::memory_start(cfg);
        let bias_weights = controls + 3 * d;
        let matrix = bias_weights + d * d;
        let frequency = matrix + d * d;
        Ok(Self {
            kernels: Kernels::load(&e.cuda)?,
            input: Frame::new(e, rows, frequency)?,
            output: Frame::new(e, rows, frequency + d / 2)?,
            ctrl: e.cuda.alloc(rows * 3)?,
            dctrl: e.cuda.alloc(rows * 3)?,
            gain: e.cuda.alloc(rows)?,
            bias: e.cuda.alloc(rows * d)?,
            dbias: e.cuda.alloc(rows * d)?,
            middle: e.cuda.alloc(rows * d)?,
            dmiddle_input: e.cuda.alloc(rows * d)?,
            final_feature: e.cuda.alloc(rows * d)?,
            dcontext: e.cuda.alloc(rows * d)?,
            controls,
            bias_weights,
            matrix,
            shape: std::cell::Cell::new(None),
        })
    }
    pub(super) fn features(&self) -> &F {
        &self.final_feature
    }

    pub(super) fn forward(&self, e: &Engine, rows: usize, length: usize) -> Result<()> {
        self.shape.set(None);
        let d = e.config.width;
        let launch = Grid::new(rows as u32, (d / 2) as u32);
        e.linear(self.ctrl.ptr(), e.last.ptr(), self.controls, rows, 3, d)?;
        e.linear(self.bias.ptr(), e.last.ptr(), self.bias_weights, rows, d, d)?;
        unsafe {
            self.kernels.positions(
                Grid::new(rows as u32, 1),
                self.input.position.ptr(),
                self.output.position.ptr(),
                self.gain.ptr(),
                self.ctrl.ptr(),
                length as u32,
            )?;
        }
        self.input.angles(e, rows)?;
        self.output.angles(e, rows)?;
        unsafe {
            self.kernels.forward::<false>(
                launch,
                self.input.rotated.ptr(),
                self.input.rotated.ptr(),
                e.embed_norm.ptr(),
                self.input.angles.ptr(),
                self.gain.ptr(),
                self.bias.ptr(),
                length as u32,
            )?;
        }
        e.linear(
            self.middle.ptr(),
            self.input.rotated.ptr(),
            self.matrix,
            rows,
            d,
            d,
        )?;
        unsafe {
            self.kernels.forward::<true>(
                launch,
                self.output.rotated.ptr(),
                self.final_feature.ptr(),
                self.middle.ptr(),
                self.output.angles.ptr(),
                self.gain.ptr(),
                self.bias.ptr(),
                length as u32,
            )?;
        }
        self.shape.set(Some((rows, length)));
        Ok(())
    }

    pub(super) fn backward(&self, e: &Engine, rows: usize, length: usize) -> Result<()> {
        if self.shape.get() != Some((rows, length)) {
            return Err("two-frame backward requires a matching forward".into());
        }
        let d = e.config.width;
        let launch = Grid::new(rows as u32, (d / 2) as u32);
        unsafe {
            self.kernels.backward::<true>(
                launch,
                self.output.dx.ptr(),
                self.output.dangles.ptr(),
                e.dlast.ptr(),
                self.output.rotated.ptr(),
                self.output.angles.ptr(),
                self.gain.ptr(),
            )?;
        }
        e.linear_back(
            self.dmiddle_input.ptr(),
            self.output.dx.ptr(),
            self.input.rotated.ptr(),
            self.matrix,
            rows,
            d,
            d,
        )?;
        unsafe {
            self.kernels.backward::<false>(
                launch,
                self.input.dx.ptr(),
                self.input.dangles.ptr(),
                self.dmiddle_input.ptr(),
                self.input.rotated.ptr(),
                self.input.angles.ptr(),
                self.gain.ptr(),
            )?;
        }
        self.input.position_back(e, rows)?;
        self.output.position_back(e, rows)?;
        unsafe {
            self.kernels.controls_back(
                Grid::new(rows as u32, 1),
                self.dctrl.ptr(),
                self.dbias.ptr(),
                self.input.dposition.ptr(),
                self.output.dposition.ptr(),
                e.dlast.ptr(),
                self.output.rotated.ptr(),
                d as u32,
                length as u32,
            )?;
        }
        e.linear_back(
            e.dlast.ptr(),
            self.dctrl.ptr(),
            e.last.ptr(),
            self.controls,
            rows,
            3,
            d,
        )?;
        e.linear_back(
            self.dcontext.ptr(),
            self.dbias.ptr(),
            e.last.ptr(),
            self.bias_weights,
            rows,
            d,
            d,
        )?;
        unsafe {
            e.ops.add(
                grid(rows * d),
                e.dlast.ptr(),
                e.dlast.ptr(),
                self.dcontext.ptr(),
                (rows * d) as u32,
            )?;
        }
        Ok(())
    }

    pub(super) fn add_input_gradient(&self, e: &Engine, target: P, rows: usize) -> Result<()> {
        let n = rows * e.config.width;
        unsafe {
            e.ops
                .add(grid(n), target, target, self.input.dx.ptr(), n as u32)?;
        }
        Ok(())
    }
}
