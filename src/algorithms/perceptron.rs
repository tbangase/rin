use derive_getters::Getters;
use typed_builder::TypedBuilder;

use crate::utils::non_zero_init_array;

use anyhow::Result;
use ndarray::{concatenate, prelude::*};

#[derive(Debug, Clone, Default, Getters, TypedBuilder)]
pub struct Perceptron {
    // 学習率: η
    #[builder(default = 0.1)]
    learning_rate: f64,
    // 訓練回数
    #[builder(default = 10)]
    train_num: u32,
    // 重み
    #[builder(default)]
    weights: Array1<f64>,
    // 誤差
    #[builder(default)]
    errors: Vec<f64>,
}

impl Perceptron {
    /// Fit (learn) with data
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        anyhow::ensure!(
            x.shape()[1] == y.len(),
            "Training Data of x and Label y has different shape."
        );

        self.weights = non_zero_init_array(1 + x.shape()[1]);
        self.errors.clear();

        // 訓練回数分のループ
        // 各訓練データで重みを更新
        for _ in 0..self.train_num {
            let mut errors = 0.;
            for (x_i, target) in x.outer_iter().zip(y.iter()) {
                let y_hat = self.predict(x_i);
                let error = target - y_hat;
                // delta_w = η * x_j * error
                for (j, weight) in self.weights.iter_mut().enumerate() {
                    if j == 0 {
                        *weight += self.learning_rate * error;
                    } else {
                        *weight += self.learning_rate * x_i[j - 1] * error;
                    }
                }
                errors += error.abs();
            }
            self.errors.push(errors);
        }
        Ok(())
    }

    /// ??
    pub fn net_input(&self, x: ArrayView1<f64>) -> f64 {
        let expanded_x = concatenate![Axis(0), array![0.], x];
        expanded_x.dot(&self.weights)
    }

    /// Predict with fitted model
    pub fn predict(&self, x: ArrayView1<f64>) -> f64 {
        if self.net_input(x) >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }
}

#[cfg(test)]
mod perceptron_test {
    use super::*;

    #[test]
    fn test_normal_perceptron() -> Result<()> {
        let mut p = Perceptron::builder().train_num(50).build();

        let x = Array2::from(vec![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let y = array![1., 1., -1.];

        p.fit(&x, &y)?;

        assert_eq!(p.predict(x.row(0)), 1.0);
        assert_eq!(p.predict(x.row(1)), 1.0);
        assert_eq!(p.predict(x.row(2)), -1.0);
        Ok(())
    }

    #[test]
    fn test_wrong_size_handling() -> Result<()> {
        let mut p = Perceptron::builder().build();

        let x = Array2::from(vec![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let y = array![1., 1., -1., -1.];

        assert!(p.fit(&x, &y).is_err());

        Ok(())
    }
}
