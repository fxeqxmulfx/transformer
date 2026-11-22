// Included in the pinned engine. These kernels accept only hidden states and
// learned coordinates. Tokens reach the backbone through its embedding lookup.
impl Engine {
    fn collapse_streams(&self, states: P, rows: usize) -> Result<()> {
        let w = self.config.width;
        let count = rows * w;
        // SAFETY: states hold four streams per row. shared is rows by width.
        unsafe {
            if let Some(read) = &self.stream_read {
                let offsets = self.parameters.at(crate::architecture::start(&self.config))
                    .ok_or("stream read outside parameter buffer")?;
                read.mix::<false>(grid(count), self.shared.ptr(), states, offsets, w as u32, count as u32)?;
            } else {
                self.mhc.as_ref().unwrap().collapse::<false>(
                    grid(count), self.shared.ptr(), states, w as u32, count as u32)?;
            }
        }
        Ok(())
    }

    fn collapse_streams_backward(&self, rows: usize) -> Result<()> {
        let w = self.config.width;
        let count = rows * w;
        // SAFETY: the engine validates at least one layer. The last output
        // and its adjoint are row-major arrays of four width-sized streams.
        unsafe {
            if let Some(read) = &self.stream_read {
                let first = crate::architecture::start(&self.config);
                let offsets = self.parameters.at(first).ok_or("stream read outside parameter buffer")?;
                let gradient = self.gradients.at(first).ok_or("stream read outside gradient buffer")?;
                read.mix::<true>(grid(count), self.stream_grad.ptr(), self.dshared.ptr(), offsets, w as u32, count as u32)?;
                read.weights_back(Grid::new((4 * w) as u32, 32), gradient,
                    self.layers.last().unwrap().out.ptr(), self.dshared.ptr(), w as u32, rows as u32)?;
            } else {
                self.mhc.as_ref().unwrap().collapse::<true>(
                    grid(count), self.stream_grad.ptr(), self.dshared.ptr(), w as u32, count as u32)?;
            }
        }
        Ok(())
    }
}
