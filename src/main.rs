use ml::Tensor;

// This binary used for exploratory testing
fn main() {
    let empty_1d: Tensor<f64>= Tensor::new_1d(vec![]);
    println!("1D empty: {:?}", empty_1d);

    let empty_2d: Tensor<f64> = Tensor::new_2d(vec![vec![]]);
    println!("2D empty: {:?}", empty_2d);

    let empty_3d: Tensor<f64> = Tensor::new(vec![0,0,0], vec![]);
    println!("3'D empty: {:?}", empty_3d);

    let scalar = Tensor::new_0d(1.0);
    println!("{}", scalar);

    let vector = Tensor::new_1d(vec![1.0, 2.0, 3.0]);
    println!("{}", vector);

    let matrix = Tensor::new_2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 4.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);
    println!("{}", matrix);
}