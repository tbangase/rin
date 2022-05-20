use ndarray::Array1;
use rand_distr::{Distribution, Normal};

pub fn non_zero_init() -> f64 {
    let normal = Normal::new(0.0, 0.01).unwrap();
    normal.sample(&mut rand::thread_rng())
}

pub fn non_zero_init_array(n: usize) -> Array1<f64> {
    let normal = Normal::new(0.0, 0.01).unwrap();
    let mut res = vec![];
    for _ in 0..n {
        res.push(normal.sample(&mut rand::thread_rng()));
    }
    Array1::from_vec(res)
}