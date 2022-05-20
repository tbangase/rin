pub mod decision_functions;
pub mod utils;

use derive_new::new;
use utils::non_zero_init_vec;
use ndarray::prelude::*;

#[derive(new, Debug, Default)]
struct Perceptron {
    // 学習率: η
    learning_rate: f64,
    // 反復回数
    iter_num: u32,
    // 重みを初期化するための乱数シード
    random_state: u32,
    // 重み
    #[new(default)]
    weights: Array1<f64>,
    #[new(default)]
    errors: Vec<usize>,
}

impl Perceptron {
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> &mut Self {
        // self.weights = non_zero_init_vec

        self
    } 
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptron() {
        let mut p = Perceptron::new(0.01, 5, 1);

        let x = Array2::from(vec![
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ]);
        let y = Array1::from_vec(vec![1., 2., 3.]);

        p.fit(&x, &y);

        assert!(true)
    } 

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
