//! Bridge from the Phasor dense backbone to its trainable slot memory.
use super::*;
use crate::slot_memory::{self, Layout, Routing, Row};
use std::cell::RefCell;

struct Tape {
    weights: Vec<f32>,
    hidden: Vec<f32>,
    rows: Vec<Row>,
    length: usize,
    routing: Routing,
}

impl Engine {
    /// Count visited dense memory vectors with nonzero learned coordinates
    /// in the latest forward. Used solely by the independent checkpoint audit.
    pub(crate) fn written_slot_reads(&self) -> Result<usize> {
        let Some(context_layer::ContextLayer::Slots(layer)) = &self.context_layer else {
            return Err("written slot reads require the slot-memory architecture".into());
        };
        let tape = layer.tape.borrow();
        Ok(tape
            .rows
            .iter()
            .filter_map(|r| r.cell)
            .filter(|&i| layer.layout.row(&tape.weights, i).iter().any(|&v| v != 0.))
            .count())
    }
}

pub(super) struct SlotLayer {
    phase: Option<super::hidden_layer::HiddenLayer>,
    layout: Layout,
    offset: usize,
    copier: crate::slot_copy::kernels::Kernels,
    addresses: crate::slot_addresses::kernels::Kernels,
    dictionary_t: F,
    inverse_norms: RefCell<F>,
    cells: Buffer<u32>,
    parameters: RefCell<F>,
    products: F,
    output: F,
    tape: RefCell<Tape>,
}

impl SlotLayer {
    pub(super) fn new(e: &Engine) -> Result<Self> {
        let layout = Layout::new(&e.config);
        let count = layout.bias + 4;
        Ok(Self {
            phase: e
                .config
                .head
                .has_triple_phase()
                .then(|| super::hidden_layer::HiddenLayer::new(e))
                .transpose()?,
            offset: crate::architecture::memory_start(&e.config),
            products: e.cuda.alloc(layout.slots.pow(3))?,
            dictionary_t: e.cuda.alloc(layout.slots * layout.d)?,
            inverse_norms: RefCell::new(e.cuda.alloc(layout.slots)?),
            cells: e.cuda.alloc(e.config.max_length * e.config.max_batch)?,
            addresses: crate::slot_addresses::kernels::Kernels::load(&e.cuda)?,
            layout,
            copier: crate::slot_copy::kernels::Kernels::load(&e.cuda)?,
            parameters: RefCell::new(e.cuda.alloc(count)?),
            output: e
                .cuda
                .alloc(e.config.max_length * e.config.max_batch * e.config.width)?,
            tape: RefCell::new(Tape {
                weights: vec![],
                hidden: vec![],
                rows: vec![],
                length: 0,
                routing: Routing::default(),
            }),
        })
    }

    fn copy(&self, dst: P, src: P, n: usize) -> Result<()> {
        // Every caller supplies two allocated ranges of at least n dense values.
        unsafe {
            self.copier.copy(grid(n), dst, src, n as u32)?;
        }
        Ok(())
    }

    pub(super) fn features(&self) -> &F {
        &self.output
    }

    pub(super) fn forward(&self, e: &Engine, rows: usize, length: usize) -> Result<()> {
        let l = &self.layout;
        let n = l.bias + 4;
        let p = e.parameters.at(self.offset).ok_or("missing slot weights")?;
        let buffer = self.parameters.borrow();
        self.copy(buffer.ptr(), p, n)?;
        let w = buffer.read()?;
        let mut tape = self.tape.borrow_mut();
        if tape.weights.get(..l.gates) != Some(&w[..l.gates]) {
            // The dictionary transpose is at most width². This dedicated
            // FP32 dot product also serves the lower-precision backbone.
            e.transpose(p, l.slots, l.d)?;
            self.copy(self.dictionary_t.ptr(), e.scratch.ptr(), l.slots * l.d)?;
            let m = l.slots * l.slots;
            let blocks = m.div_ceil(gemm::kernels::BM) * l.slots.div_ceil(gemm::kernels::BN);
            let values = e
                .parameters
                .at(self.offset + l.values)
                .ok_or("missing slot values")?;
            unsafe {
                e.gemm.sgemm::<false>(
                    Grid::new(blocks as u32, gemm::kernels::THREADS as u32),
                    self.products.ptr(),
                    values,
                    e.scratch.ptr(),
                    m as u32,
                    l.slots as u32,
                    l.d as u32,
                )?;
            }
            tape.routing.refresh(l, &w, &self.products.read()?);
            self.inverse_norms
                .borrow_mut()
                .write(tape.routing.inverse_norms())?;
        }
        // A block sees just its row and the preceding row of this sequence.
        // Both scores for every slot fit in the block's shared memory.
        unsafe {
            self.addresses.cells(
                Grid::new(rows as u32, l.slots as u32),
                self.cells.ptr(),
                e.embed_norm.ptr(),
                e.last.ptr(),
                self.dictionary_t.ptr(),
                self.inverse_norms.borrow().ptr(),
                l.d as u32,
                length as u32,
                (l.d as f32).sqrt().recip(),
            )?;
        }
        let cells = self.cells.read()?;
        let mut h = e.last.read()?;
        h.truncate(rows * l.d);
        let (choices, out) = slot_memory::forward(l, &w, &tape.routing, &cells[..rows], &h, length);
        let device = e.cuda.upload(&out)?;
        self.copy(self.output.ptr(), device.ptr(), out.len())?;
        if let Some(phase) = &self.phase {
            phase.forward_features(e, rows, length)?;
            unsafe {
                e.ops.add(
                    grid(out.len()),
                    self.output.ptr(),
                    self.output.ptr(),
                    phase.features().ptr(),
                    out.len() as u32,
                )?;
            }
        }
        tape.weights = w;
        tape.hidden = h;
        tape.rows = choices;
        tape.length = length;
        Ok(())
    }

    pub(super) fn backward(&self, e: &Engine, rows: usize, length: usize) -> Result<()> {
        let tape = self.tape.borrow();
        if tape.rows.len() != rows || tape.length != length {
            return Err("slot backward needs a matching dense forward".into());
        }
        if let Some(phase) = &self.phase {
            phase.backward(e, rows, length)?;
        }
        let mut buffer = self.parameters.borrow_mut();
        let mut dw = vec![0.; buffer.len()];
        let dy = e.dlast.read()?;
        let dh = slot_memory::backward(
            &self.layout,
            &tape.weights,
            &tape.hidden,
            &tape.rows,
            &dy,
            length,
            &mut dw,
        );
        buffer.write(&dw)?;
        self.copy(
            e.gradients
                .at(self.offset)
                .ok_or("missing slot gradients")?,
            buffer.ptr(),
            dw.len(),
        )?;
        let device = e.cuda.upload(&dh)?;
        self.copy(e.dlast.ptr(), device.ptr(), dh.len())
    }

    pub(super) fn add_input_gradient(&self, e: &Engine, target: P, rows: usize) -> Result<()> {
        if let Some(phase) = &self.phase {
            phase.add_projection_gradient(e, target, rows)?;
        }
        Ok(())
    }
}
