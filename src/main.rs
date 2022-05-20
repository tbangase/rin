extern crate ndarray;
use std::time::Instant;

use ndarray::prelude::*;
use rin::{utils::non_zero_init_array, Perceptron};

fn main() -> anyhow::Result<()> {
    std::env::set_var("RUST_LOG", "debug");
    tracing_subscriber::fmt::init();
    let start = Instant::now();

    let vector = vec![1., 2., 3.];

    // Not good for definition
    let a = Array::from_vec(
        vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8],
        ]
    );

    // Accurate definition
    let b = Array::from_shape_vec(
        (2, 2),
        vec![1, 2, 3, 4]
    );

    // Easiest Definition
    let c = array![
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ];

    let x = arr2(&[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]);

    let array1 = Array1::from_vec(vector);
    let non_zero_array = non_zero_init_array(3);

    tracing::debug!("\n{:?}", a);
    tracing::debug!("\n{:?}", b);
    tracing::debug!("\n{:?}", c);
    tracing::debug!("\n{:?}", x);
    tracing::debug!("\n{:?}", array1);
    tracing::debug!("\n{:?}", non_zero_array);


    let mut p = Perceptron::new()
        .set_learning_rate(0.01)
        .set_train_num(100)
        .clone();

    let x = arr2(&[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]);

    let y = arr1(&[1.0, 1.0, -1.0]);

    p.fit(&x, &y)?;

    tracing::debug!("\nExpect as  1.0: {:?}", p.predict(x.row(0)));
    tracing::debug!("\nExpect as  1.0: {:?}", p.predict(x.row(1)));
    tracing::debug!("\nExpect as -1.0: {:?}", p.predict(x.row(2)));

    tracing::debug!("\n{:?}", p.weights());

    let duration = start.elapsed();
    tracing::info!("Total Duration: {:?}", duration);
    Ok(())
}