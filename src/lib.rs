pub mod decision_functions;
pub mod utils;

use derive_new::new;
use utils::non_zero_init_array;
use ndarray::{prelude::*, concatenate};
use getset::{Getters, Setters};

#[derive(new, Debug, Clone, Default, Getters, Setters)]
pub struct Perceptron {
    // 学習率: η
    #[new(value = "0.1")]
    #[getset(get = "pub", set = "pub")]
    learning_rate: f64,
    // 訓練回数
    #[new(value = "50")]
    #[getset(get = "pub", set = "pub")]
    train_num: u32,
    // 重み
    #[new(default)]
    #[getset(get = "pub")]
    weights: Array1<f64>,
    // 誤差
    #[new(default)]
    #[getset(get = "pub")]
    errors: Vec<f64>,
}

impl Perceptron {
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> &mut Self {
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

        self
    } 

    pub fn net_input(&self, x: ArrayView1<f64>) -> f64 {
        let expanded_x = concatenate![Axis(0), array![0.], x];
        println!("Expanded X: {expanded_x:?}\n");
        expanded_x.dot(&self.weights)
    }

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
    fn test_normal_perceptron() {
        let mut p = Perceptron::new();

        let x = Array2::from(vec![
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ]);
        let y = array![1., 1., -1.];

        p.fit(&x, &y);

        assert_eq!(p.predict(x.row(0)),  1.0); 
        assert_eq!(p.predict(x.row(1)),  1.0); 
        assert_eq!(p.predict(x.row(2)), -1.0); 

    } 

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
