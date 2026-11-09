// Retain task #188's source instrumentation at 4b8d18e9 without changing it.
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../single_update/src/instrumentation.rs"
));

impl Engine {
    /// Expose existing moment buffers to the unchanged GPT optimizer kernel.
    pub fn wsd_moments(&self) -> (kernite_cuda::Ptr<f32>, kernite_cuda::Ptr<f32>) {
        (self.first.ptr(), self.second.ptr())
    }
}
