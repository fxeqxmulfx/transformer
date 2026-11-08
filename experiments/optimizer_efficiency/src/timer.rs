//! CUDA driver event API, matching NVIDIA's official signatures:
//! https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html
//! Events bracket queued work on Kernite's default stream. Spans include
//! possible queue gaps; host wall time also charges launches and completion.
use crate::{domain::Result, inputs::require};
use kernite_cuda::Cuda;
use serde_json::{Value, json};
use std::{
    ffi::{CStr, c_char, c_int, c_void},
    ptr,
    time::Instant,
};
type Handle = *mut c_void;
type Create = unsafe extern "C" fn(*mut Handle, u32) -> c_int;
type Record = unsafe extern "C" fn(Handle, Handle) -> c_int;
type One = unsafe extern "C" fn(Handle) -> c_int;
type Elapsed = unsafe extern "C" fn(*mut f32, Handle, Handle) -> c_int;
unsafe extern "C" {
    fn dlopen(file: *const c_char, flags: c_int) -> Handle;
    fn dlsym(handle: Handle, symbol: *const c_char) -> Handle;
    fn dlclose(handle: Handle) -> c_int;
}
fn check(code: c_int, operation: &str) -> Result<()> {
    require(code == 0, &format!("{operation}: CUDA result {code}"))
}
pub struct Events<'a> {
    cuda: &'a Cuda,
    library: Handle,
    start: Handle,
    end: Handle,
    record: Record,
    synchronize: One,
    elapsed: Elapsed,
    destroy: One,
}
impl<'a> Events<'a> {
    pub fn new(cuda: &'a Cuda) -> Result<Self> {
        cuda.sync()?;
        // SAFETY: Linux driver library and CUDA-documented C function types.
        let library = unsafe { dlopen(c"libcuda.so.1".as_ptr(), 2) };
        require(!library.is_null(), "cannot open CUDA event API")?;
        let load = |name: &CStr| -> Result<Handle> {
            let address = unsafe { dlsym(library, name.as_ptr()) };
            require(!address.is_null(), "missing CUDA event symbol")?;
            Ok(address)
        };
        let outcome = (|| -> Result<Self> {
            let create = unsafe { std::mem::transmute::<Handle, Create>(load(c"cuEventCreate")?) };
            let record = unsafe { std::mem::transmute::<Handle, Record>(load(c"cuEventRecord")?) };
            let synchronize =
                unsafe { std::mem::transmute::<Handle, One>(load(c"cuEventSynchronize")?) };
            let elapsed =
                unsafe { std::mem::transmute::<Handle, Elapsed>(load(c"cuEventElapsedTime")?) };
            let destroy =
                unsafe { std::mem::transmute::<Handle, One>(load(c"cuEventDestroy_v2")?) };
            let mut start = ptr::null_mut();
            let mut end = ptr::null_mut();
            check(unsafe { create(&mut start, 0) }, "cuEventCreate start")?;
            if let Err(error) = check(unsafe { create(&mut end, 0) }, "cuEventCreate end") {
                unsafe { destroy(start) };
                return Err(error);
            }
            Ok(Self {
                cuda,
                library,
                start,
                end,
                record,
                synchronize,
                elapsed,
                destroy,
            })
        })();
        if outcome.is_err() {
            unsafe { dlclose(library) };
        }
        outcome
    }
    pub fn measure(&self, work: impl FnOnce() -> Result<()>) -> Result<Value> {
        self.cuda.sync()?;
        let started = Instant::now();
        check(
            unsafe { (self.record)(self.start, ptr::null_mut()) },
            "cuEventRecord start",
        )?;
        work()?;
        check(
            unsafe { (self.record)(self.end, ptr::null_mut()) },
            "cuEventRecord end",
        )?;
        check(
            unsafe { (self.synchronize)(self.end) },
            "cuEventSynchronize",
        )?;
        let wall = started.elapsed().as_secs_f64();
        let mut ms = 0f32;
        check(
            unsafe { (self.elapsed)(&mut ms, self.start, self.end) },
            "cuEventElapsedTime",
        )?;
        require(ms.is_finite() && ms >= 0., "invalid event time")?;
        Ok(json!({"wall_seconds":wall,"gpu_seconds":ms as f64/1000.}))
    }
}
impl Drop for Events<'_> {
    fn drop(&mut self) {
        // SAFETY: these events belong to the borrowed live CUDA context;
        // destruction is valid even if a failed workload remains pending.
        unsafe {
            (self.destroy)(self.start);
            (self.destroy)(self.end);
            dlclose(self.library);
        }
    }
}
pub fn stats(values: impl IntoIterator<Item = f64>) -> Value {
    let mut values: Vec<_> = values.into_iter().collect();
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return Value::Null;
    }
    json!({"median":values[values.len()/2],"p90":values[(9*values.len()).div_ceil(10)-1],"min":values[0],"max":values[values.len()-1]})
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nearest_rank_retains_the_slow_tail() {
        let s = stats([9., 1., 7., 2., 6., 3., 5., 4., 100.]);
        assert_eq!(s["median"], 5.);
        assert_eq!(s["p90"], 100.);
        assert!(stats([]).is_null());
    }
}
