use rand_distr::{Distribution, Normal};

pub fn non_zero_init() -> f64 {
    let normal = Normal::new(0.0, 0.01).unwrap();
    normal.sample(&mut rand::thread_rng())
}

pub fn non_zero_init_vec(n: usize) -> Vec<f64> {
    let normal = Normal::new(0.0, 0.01).unwrap();
    let mut res = vec![];
    for _ in 0..n {
        res.push(normal.sample(&mut rand::thread_rng()));
    }
    res
}