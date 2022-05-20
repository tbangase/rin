extern crate ndarray;
use std::time::Instant;

use ndarray::prelude::*;

fn main() {
    let start = Instant::now();

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

    println!("{:?}", a);
    println!("{:?}", b);
    println!("{:?}", c);
    println!("{:?}", x);

    let duration = start.elapsed();
    println!("Rand Init vec Duration: {:?}", duration);
}