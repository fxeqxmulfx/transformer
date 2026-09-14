use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Tensor, TensorData};

pub type B = NdArray<f64, i64, i8>;

pub fn probe() -> f64 {
    let d = NdArrayDevice::Cpu;
    let a = Tensor::<B, 2>::from_data(TensorData::from([[1.0f64, 2.0], [3.0, 4.0]]), &d);
    let b = Tensor::<B, 2>::from_data(TensorData::from([[1.0f64], [1.0]]), &d);
    let c = a.matmul(b);
    c.into_data().to_vec::<f64>().unwrap()[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn f64_is_really_f64() {
        assert_eq!(probe(), 3.0);
        let d = NdArrayDevice::Cpu;
        // 2^53 - 1 survives a matmul only if the element type is f64.
        let big = 9007199254740991.0f64;
        let a = Tensor::<B, 2>::from_data(TensorData::from([[big]]), &d);
        let b = Tensor::<B, 2>::from_data(TensorData::from([[1.0f64]]), &d);
        assert_eq!(a.matmul(b).into_data().to_vec::<f64>().unwrap()[0], big);
    }
}
