//! Full latent DFT, ported from convex_stress_moments/src/fft.rs at 7d299b04.
//! Input side and latent dimension are independent here; both odd and even
//! latent dimensions retain all states. No task metadata or target rule is read.
//! CPU FFTW3 API: https://www.fftw.org/fftw3_doc/Advanced-Complex-DFTs.html
//! Only execution is thread safe; serialize plan construction and destruction.
use crate::{C, Result};
use std::{
    ffi::{c_int, c_uint, c_void},
    sync::Mutex,
};

static PLANNER: Mutex<()> = Mutex::new(());

#[link(name = "libfftw3.so.3", kind = "dylib", modifiers = "+verbatim")]
unsafe extern "C" {
    fn fftw_plan_many_dft(
        rank: c_int,
        n: *const c_int,
        howmany: c_int,
        input: *mut C,
        inembed: *const c_int,
        istride: c_int,
        idist: c_int,
        output: *mut C,
        onembed: *const c_int,
        ostride: c_int,
        odist: c_int,
        sign: c_int,
        flags: c_uint,
    ) -> *mut c_void;
    fn fftw_execute(plan: *mut c_void);
    fn fftw_destroy_plan(plan: *mut c_void);
}

pub struct Fourier {
    n: usize,
    input: Vec<C>,
    output: Vec<C>,
    forward: *mut c_void,
    backward: *mut c_void,
}
impl Fourier {
    pub fn new(side: usize, n: usize) -> Result<Self> {
        if !(2..=256).contains(&side) || !(2..=512).contains(&n) {
            return Err("unsupported input side or latent Fourier dimension".into());
        }
        let mut out = Self {
            n,
            input: vec![C::default(); side * side * n],
            output: vec![C::default(); side * side * n],
            forward: std::ptr::null_mut(),
            backward: std::ptr::null_mut(),
        };
        let _guard = PLANNER.lock().map_err(|_| "FFTW planner lock poisoned")?;
        let size = n as c_int;
        let cells = (side * side) as c_int;
        // These are CPU fftw3.h flags, not the different CUDA cufftw.h values.
        let flags = (1 << 6) | (1 << 1); // ESTIMATE | UNALIGNED
        // The owned buffers never move or resize. Class-contiguous real inputs
        // become frequency-major column-major matrices. Null embeds mean length n.
        unsafe {
            out.forward = fftw_plan_many_dft(
                1,
                &size,
                cells,
                out.input.as_mut_ptr(),
                std::ptr::null(),
                1,
                size,
                out.output.as_mut_ptr(),
                std::ptr::null(),
                cells,
                1,
                1,
                flags,
            );
            out.backward = fftw_plan_many_dft(
                1,
                &size,
                cells,
                out.output.as_mut_ptr(),
                std::ptr::null(),
                cells,
                1,
                out.input.as_mut_ptr(),
                std::ptr::null(),
                1,
                size,
                -1,
                flags,
            );
        }
        drop(_guard);
        if out.forward.is_null() || out.backward.is_null() {
            return Err("FFTW plan construction failed".into());
        }
        Ok(out)
    }
    pub fn forward(&mut self, p: &[f64]) -> Result<Vec<C>> {
        if p.len() != self.input.len() || p.iter().any(|x| !x.is_finite()) {
            return Err("invalid Fourier input".into());
        }
        for (z, &x) in self.input.iter_mut().zip(p) {
            *z = C::new(x, 0.);
        }
        // The plan uses exactly the live, disjoint owned buffers above.
        unsafe { fftw_execute(self.forward) };
        if self
            .output
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err("Fourier transform overflow".into());
        }
        Ok(self.output.clone())
    }
    pub fn inverse(&mut self, moments: &[C]) -> Result<Vec<f64>> {
        if moments.len() != self.output.len()
            || moments
                .iter()
                .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err("invalid inverse Fourier input".into());
        }
        self.output.copy_from_slice(moments);
        unsafe { fftw_execute(self.backward) };
        if self
            .input
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err("inverse Fourier transform overflow".into());
        }
        let scale = self.input.iter().map(|z| z.re.abs()).fold(1., f64::max);
        if self.input.iter().any(|z| z.im.abs() > 1e-10 * scale) {
            return Err("moments lost conjugate symmetry".into());
        }
        Ok(self.input.iter().map(|z| z.re / self.n as f64).collect())
    }
}
impl Drop for Fourier {
    fn drop(&mut self) {
        let _guard = PLANNER.lock().unwrap_or_else(|e| e.into_inner());
        // Destroy plans before their buffers. No process-global cleanup is used.
        unsafe {
            if !self.forward.is_null() {
                fftw_destroy_plan(self.forward);
            }
            if !self.backward.is_null() {
                fftw_destroy_plan(self.backward);
            }
        }
    }
}
