//! Checked prefix transfers using the pinned kernite-cuda driver signatures.
//! Buffer::read/write in a1b6bc2 require the whole allocation; serving must
//! transfer only its request, including a compacted partial fallback batch.
use crate::{domain::Result, inputs::require};
use kernite_cuda::{Buffer, Cuda, Plain};
use std::ffi::{c_char, c_int, c_void};
type Handle = *mut c_void;
type Upload = unsafe extern "C" fn(u64, *const c_void, usize) -> c_int;
type Download = unsafe extern "C" fn(*mut c_void, u64, usize) -> c_int;
unsafe extern "C" {
    fn dlopen(name: *const c_char, flags: c_int) -> Handle;
    fn dlsym(handle: Handle, name: *const c_char) -> Handle;
    fn dlclose(handle: Handle) -> c_int;
}
pub struct Transfers<'a> {
    _cuda: &'a Cuda,
    library: Handle,
    upload: Upload,
    download: Download,
}
impl<'a> Transfers<'a> {
    pub fn new(cuda: &'a Cuda) -> Result<Self> {
        cuda.sync()?;
        // SAFETY: signatures match crates/cuda/src/lib.rs at the pinned revision.
        let library = unsafe { dlopen(c"libcuda.so.1".as_ptr(), 2) };
        require(!library.is_null(), "CUDA transfer library unavailable")?;
        let up = unsafe { dlsym(library, c"cuMemcpyHtoD_v2".as_ptr()) };
        let down = unsafe { dlsym(library, c"cuMemcpyDtoH_v2".as_ptr()) };
        if up.is_null() || down.is_null() {
            unsafe { dlclose(library) };
            return Err("CUDA transfer symbols unavailable".into());
        }
        Ok(Self {
            _cuda: cuda,
            library,
            upload: unsafe { std::mem::transmute::<Handle, Upload>(up) },
            download: unsafe { std::mem::transmute::<Handle, Download>(down) },
        })
    }
    pub fn write<T: Plain>(&self, buffer: &mut Buffer<T>, data: &[T]) -> Result<()> {
        require(data.len() <= buffer.len(), "transfer exceeds allocation")?;
        // SAFETY: buffers and the borrowed CUDA context remain live on this
        // thread and device; checked lengths bound the synchronous copy.
        let code = unsafe {
            (self.upload)(
                buffer.ptr().addr(),
                data.as_ptr().cast(),
                std::mem::size_of_val(data),
            )
        };
        require(code == 0, &format!("cuMemcpyHtoD_v2: {code}"))
    }
    pub fn read<T: Plain>(&self, buffer: &Buffer<T>, count: usize) -> Result<Vec<T>> {
        require(count <= buffer.len(), "transfer exceeds allocation")?;
        let mut out = vec![T::default(); count];
        let code = unsafe {
            (self.download)(
                out.as_mut_ptr().cast(),
                buffer.ptr().addr(),
                std::mem::size_of_val(out.as_slice()),
            )
        };
        require(code == 0, &format!("cuMemcpyDtoH_v2: {code}"))?;
        Ok(out)
    }
}
impl Drop for Transfers<'_> {
    fn drop(&mut self) {
        unsafe { dlclose(self.library) };
    }
}
