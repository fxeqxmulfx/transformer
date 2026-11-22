//! The current value uses normalized embeddings and the operator uses strict prefix context.
use super::*;
use crate::toeplitz::kernels::Kernels;

struct Control {
    values: F,
    gradient: F,
    offset: usize,
    channels: usize,
}

impl Control {
    fn new(e: &Engine, rows: usize, channels: usize, offset: usize) -> Result<Self> {
        Ok(Self {
            values: e.cuda.alloc(rows * channels)?,
            gradient: e.cuda.alloc(rows * channels)?,
            offset,
            channels,
        })
    }
    fn forward(&self, e: &Engine, rows: usize) -> Result<()> {
        e.linear(
            self.values.ptr(),
            e.last.ptr(),
            self.offset,
            rows,
            self.channels,
            e.config.width,
        )
    }
    fn backward(&self, e: &Engine, dx: P, rows: usize) -> Result<()> {
        e.linear_back(
            dx,
            self.gradient.ptr(),
            e.last.ptr(),
            self.offset,
            rows,
            self.channels,
            e.config.width,
        )
    }
}

pub(super) struct ToeplitzLayer {
    kernels: Kernels,
    stencil: Control,
    gain: Control,
    bias: Control,
    mixed: F,
    dmixed: F,
    dcontext: F,
}

impl ToeplitzLayer {
    pub(super) fn new(e: &Engine, rows: usize) -> Result<Self> {
        let d = e.config.width;
        let projections = if e.config.head == crate::head::Head::TiedToeplitz {
            2
        } else {
            4
        };
        let controls = crate::architecture::memory_start(&e.config) + projections * d * d;
        let gain = controls + (2 * d - 1) * d;
        let bias = gain + 2 * d * d;
        Ok(Self {
            kernels: Kernels::load(&e.cuda)?,
            stencil: Control::new(e, rows, 2 * d - 1, controls)?,
            gain: Control::new(e, rows, 2 * d, gain)?,
            bias: Control::new(e, rows, 2 * d, bias)?,
            mixed: e.cuda.alloc(rows * 2 * d)?,
            dmixed: e.cuda.alloc(rows * 2 * d)?,
            dcontext: e.cuda.alloc(rows * d)?,
        })
    }

    pub(super) fn forward(
        &self,
        e: &Engine,
        read: P,
        value: P,
        rows: usize,
        length: usize,
    ) -> Result<()> {
        for control in [&self.stencil, &self.gain, &self.bias] {
            control.forward(e, rows)?;
        }
        let d = e.config.width as u32;
        unsafe {
            self.kernels.forward(
                Grid::new(rows as u32, d),
                read,
                self.mixed.ptr(),
                value,
                self.stencil.values.ptr(),
                self.gain.values.ptr(),
                self.bias.values.ptr(),
                d,
                length as u32,
            )?;
        }
        Ok(())
    }

    pub(super) fn backward(
        &self,
        e: &Engine,
        dvalue: P,
        value: P,
        dread: P,
        rows: usize,
        length: usize,
    ) -> Result<()> {
        let d = e.config.width;
        let launch = Grid::new(rows as u32, d as u32);
        let length = length as u32;
        unsafe {
            self.kernels.diagonal_back(
                launch,
                self.dmixed.ptr(),
                self.gain.gradient.ptr(),
                self.bias.gradient.ptr(),
                dread,
                self.mixed.ptr(),
                self.gain.values.ptr(),
                d as u32,
                length,
            )?;
            self.kernels.value_back(
                launch,
                dvalue,
                self.dmixed.ptr(),
                self.stencil.values.ptr(),
                d as u32,
                length,
            )?;
            self.kernels.stencil_back(
                Grid::new(rows as u32, (2 * d - 1) as u32),
                self.stencil.gradient.ptr(),
                self.dmixed.ptr(),
                value,
                d as u32,
                length,
            )?;
        }
        self.stencil.backward(e, e.dlast.ptr(), rows)?;
        for control in [&self.gain, &self.bias] {
            control.backward(e, self.dcontext.ptr(), rows)?;
            unsafe {
                e.ops.add(
                    grid(rows * d),
                    e.dlast.ptr(),
                    e.dlast.ptr(),
                    self.dcontext.ptr(),
                    (rows * d) as u32,
                )?;
            }
        }
        Ok(())
    }
}
