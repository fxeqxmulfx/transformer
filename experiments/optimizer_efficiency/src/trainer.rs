//! Both variants use the unchanged source forward and full backward. Source
//! snapshots are canonical even when device moments use independent owners.
use crate::{
    application::{Backend, Metrics, State},
    domain::{
        Result,
        data::{Data, Plan},
        model::Arm,
    },
    fast_kernel,
    infrastructure::engine::Engine,
    inputs::require,
    layout, state,
};
use kernite_cuda::Cuda;
use std::cell::Cell;
pub struct Trainer {
    engine: Engine,
    fast: Option<fast_kernel::kernels::Kernels>,
    initial_zero: Cell<bool>,
}
impl Trainer {
    pub fn new(
        arm: Arm,
        p: &[f32],
        data: &Data,
        plan: &Plan,
        compact: bool,
        cuda: &Cuda,
    ) -> Result<Self> {
        let mut engine = Engine::new(arm, p, data, plan)?;
        let fast = if compact {
            engine.compact_moments()?;
            Some(fast_kernel::kernels::Kernels::load(cuda)?)
        } else {
            None
        };
        Ok(Self {
            engine,
            fast,
            initial_zero: Cell::new(true),
        })
    }
    pub fn arm(&self) -> Arm {
        self.engine.arm
    }
    pub fn name(&self) -> &'static str {
        if self.fast.is_some() {
            "compact-adamw"
        } else {
            "source-adamw"
        }
    }
    pub fn bytes(&self) -> usize {
        self.engine.allocated_bytes()
    }
    pub fn moment_bytes(&self) -> usize {
        self.engine.moment_bytes()
    }
    pub fn prepare_gradient(&self, offset: usize, batch: usize) -> Result<Vec<f32>> {
        self.engine.forward(false, offset, batch, true)?;
        self.engine.backward(batch)?;
        Ok(self.engine.gradients.read()?)
    }
    pub fn set_gradient(&mut self, g: &[f32]) -> Result<()> {
        self.engine.gradients.write(g)?;
        Ok(())
    }
    pub fn update(&self, step: usize) -> Result<()> {
        if let Some(fast) = &self.fast {
            self.engine.compact_update(step, fast)?;
            self.initial_zero.set(false);
            Ok(())
        } else {
            self.engine.update(step)
        }
    }
}
impl Backend for Trainer {
    fn train_step(&self, offset: usize, batch: usize, step: usize) -> Result<()> {
        if self.fast.is_none() {
            return self.engine.train_step(offset, batch, step);
        }
        self.engine.forward(false, offset, batch, true)?;
        self.engine.backward(batch)?;
        self.update(step)
    }
    fn evaluate(&self, validation: bool) -> Result<Metrics> {
        self.engine.evaluate(validation)
    }
    fn state(&self) -> Result<State> {
        let mut state = self.engine.state()?;
        if self.fast.is_some() && self.arm() != Arm::Swiglu {
            state.first = if self.initial_zero.get() {
                vec![0.; self.arm().count()]
            } else {
                layout::expand(self.arm(), &state.first, false)
            };
            state.second = layout::expand(self.arm(), &state.second, true);
        }
        Ok(state)
    }
    fn restore(&mut self, value: &State) -> Result<()> {
        require(state::valid(self.arm(), value), "invalid canonical state")?;
        if self.fast.is_some() && self.arm() != Arm::Swiglu {
            // Source allocation is all +0 initially, including aliases whose
            // signed reconstruction would produce -0. Preserve that exact
            // canonical representation until the first source-equivalent write.
            let initial = value.first.iter().all(|v| v.to_bits() == 0);
            require(
                initial
                    || layout::bits(
                        &layout::expand(self.arm(), &layout::pack(self.arm(), &value.first), false),
                        &value.first,
                    ),
                "first-moment aliases cannot round-trip bitwise",
            )?;
            require(
                layout::bits(
                    &layout::expand(self.arm(), &layout::pack(self.arm(), &value.second), true),
                    &value.second,
                ),
                "second-moment aliases cannot round-trip bitwise",
            )?;
            self.engine.restore_compact(value)?;
            self.initial_zero.set(initial);
            Ok(())
        } else {
            self.engine.restore(value)
        }
    }
}
