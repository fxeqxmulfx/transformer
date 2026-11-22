//! Context transforms produce the final dense feature and return gradients along both input paths.
use super::*;
use super::{context_rotation_layer::ContextRotation, two_frame_layer::TwoFrame};

pub(super) enum ContextLayer {
    Rotation(ContextRotation),
    Frames(TwoFrame),
}

impl ContextLayer {
    pub(super) fn new(e: &Engine) -> Result<Self> {
        if e.config.head == crate::head::Head::TwoFrame {
            Ok(Self::Frames(TwoFrame::new(e)?))
        } else {
            Ok(Self::Rotation(ContextRotation::new(e)?))
        }
    }
    pub(super) fn features(&self) -> &F {
        match self {
            Self::Rotation(x) => x.features(),
            Self::Frames(x) => x.features(),
        }
    }
    pub(super) fn forward(&self, e: &Engine, rows: usize, length: usize) -> Result<()> {
        match self {
            Self::Rotation(x) => x.forward(e, rows, length),
            Self::Frames(x) => x.forward(e, rows, length),
        }
    }
    pub(super) fn backward(&self, e: &Engine, rows: usize, length: usize) -> Result<()> {
        match self {
            Self::Rotation(x) => x.backward(e, rows, length),
            Self::Frames(x) => x.backward(e, rows, length),
        }
    }
    pub(super) fn add_input_gradient(&self, e: &Engine, target: P, rows: usize) -> Result<()> {
        match self {
            Self::Rotation(x) => x.add_input_gradient(e, target, rows),
            Self::Frames(x) => x.add_input_gradient(e, target, rows),
        }
    }
}
