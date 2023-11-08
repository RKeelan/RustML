
use std::fmt::{Debug, Display, Formatter, Result};

use derivative::Derivative;
use num_traits::Float;


pub trait Flt: Float + Display + Debug {}
impl<T: Float + Display + Debug> Flt for T {}

#[derive(Derivative)]
#[derivative(Debug)]
pub struct Tensor<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T> Tensor<T> {
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Tensor<T> {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        Tensor {
            shape,
            data,
        }
    }

    pub fn new_0d(data: T) -> Tensor<T> {
        Tensor {
            shape: vec![1],
            data: vec![data],
        }
    }

    pub fn new_1d(data: Vec<T>) -> Tensor<T> {
        Tensor {
            shape: vec![data.len()],
            data,
        }
    }

    pub fn new_2d(data: Vec<Vec<T>>) -> Tensor<T> {
        // Check for jaggedness
        let row_len = data.iter().map(|v| v.len()).collect::<Vec<usize>>();
        assert_eq!(row_len.iter().all(|v| v == &row_len[0]), true);

        let shape = match row_len[0] {
            0 => vec![0, 0],
            _ => data.iter().map(|v| v.len()).collect()
        };

        Tensor {
            shape: shape,
            data: data.into_iter().flatten().collect(),
        }
    }
}

impl<T: Flt> Display for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.shape.len() {
            1 => { write!(f, "{:?}", self.data) }
            2 => {
                let cols = self.shape[1];
                let rows = self.shape[0];
                write!(f, "Shape: {:?} ", self.shape)?;
                for i in 0..rows {
                    let row = &self.data[i * cols..(i + 1) * cols];
                    write!(f, "{:?}", row)?;
                }
                write!(f, "")
            }
            // TODO: I'd like to handle this more nicely, by printing the tensor in 2D blocks, the way PyTorch does
            _ => { write!(f, "{:?}: {:?}", self.shape, self.data) }
        }
    }
}


#[cfg(test)]
mod shape_tests {
    use crate::Tensor;

    #[test]
    fn new() {
        let empty: Tensor<f64>= Tensor::new(vec![0,0,0], vec![]);
        assert_eq!(empty.shape, vec![0,0,0]);
        assert_eq!(empty.data.len(), 0);

        // 0D
        let _ = Tensor::new(vec![1], vec![1.0]);

        // 1D
        let _ = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);

        // 2D
        let _ = Tensor::new(vec![2,3], vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        ]);
        let _ = Tensor::new(vec![1,3,2], vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        ]);

        // 3D
        let _ = Tensor::new(vec![2,3,2], vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        ]);
        
        // 4D
        let _ = Tensor::new(vec![2,2,2,2], vec![
            // Assuming NCHW,
            // Batch 0, Channel 0
            1.0, 2.0,
            3.0, 4.0,
            
            // Batch 0, Channel 1
            5.0, 6.0,
            7.0, 8.0,
            
            // Batch 1, Channel 0
            9.0, 10.0,
            11.0, 12.0,
            
            // Batch 1, Channel 1
            13.0, 14.0,
            15.0, 16.0
        ]);

        // TODO: Add more--for example, what about shape: [1,0,1]?
    }
    #[test]
    #[should_panic(expected = "")]
    fn new_invalid_shape() {
        let _ = Tensor::new(vec![2], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn new_1d() {
        let empty: Tensor<f32> = Tensor::new_1d(vec![]);
        assert_eq!(empty.shape, vec![0]);
        assert_eq!(empty.data.len(), 0);
        let _ = Tensor::new_1d(vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn new_2d() {
        let empty: Tensor<f32> = Tensor::new_2d(vec![vec![]]);
        assert_eq!(empty.shape, vec![0,0]);
        assert_eq!(empty.data.len(), 0);
        let _ = Tensor::new_2d(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0]]
        );
    }
    #[test]
    #[should_panic(expected = "")]
    fn new_2d_jagged() {
        let _ = Tensor::new_2d(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0]]
        );
    }
}