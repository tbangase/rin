mod simple_perceptron;

pub use simple_perceptron::*;

use ndarray::{Array1, Array2, ArrayView1};

pub trait Perceptron {
    /// Fit (learn) with data
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> anyhow::Result<()>;

    /// 総入力の計算 TODO: Consider this function need to public or not.
    fn net_input(&self, x: ArrayView1<f64>) -> f64;

    /// Predict with fitted model
    fn predict(&self, x: ArrayView1<f64>) -> f64;
}
