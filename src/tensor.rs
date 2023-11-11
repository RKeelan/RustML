
use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::Add;

use derivative::Derivative;

use crate::Dtype;
use crate::TensorError;


// This is inspired by PyTorch's no_grad(), and is definitely not idiomatic Rust. If this were production code, I'd
// probably just bite the bullet and make grad and no_grad versions of the constructors. But right now, it's just a
// learning project, so I'm taking the lazy route.
pub static mut REQUIRES_GRAD: bool = false;

// TODO: Following PyTorch, disallow gradients for integer-based tensors

#[derive(Derivative)]
#[derivative(Debug)]
pub struct Tensor<'a, T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
    pub grad: Option<RefCell<Vec<T>>>,
    requires_grad: bool,
    #[derivative(Debug="ignore")]
    back_prop_fn: Option<fn(result: &Tensor<'a, T>, ctx: &BackPropCtx<'a, T>)>,
    back_prop_ctx: Option<BackPropCtx<'a, T>>,
}

// Constructors
impl<'a, T: Dtype> Tensor<'a, T> {
    pub fn new_0d(data: T) -> Tensor<'a, T> {
        unsafe {
            let grad = if REQUIRES_GRAD {
                Some(RefCell::new(vec![T::zero(); 1]))
            }
            else {
                None
            };
            Tensor {
                shape: vec![1],
                data: vec![data],
                grad: grad,
                requires_grad: REQUIRES_GRAD,
                back_prop_fn: None,
                back_prop_ctx: None,
            }
        }
    }

    pub fn new_1d(data: Vec<T>) -> Tensor<'a, T> {
        let length = data.len();
        unsafe {
            let grad = if REQUIRES_GRAD {
                Some(RefCell::new(vec![T::zero(); length]))
            }
            else {
                None
            };
            Tensor {
                shape: vec![data.len()],
                data,
                grad: grad,
                requires_grad: REQUIRES_GRAD,
                back_prop_fn: None,
                back_prop_ctx: None,
            }
        }
    }

    pub fn new_2d(data: Vec<Vec<T>>) -> Tensor<'a, T> {
        // Check for jaggedness
        let row_len = data.iter().map(|v| v.len()).collect::<Vec<usize>>();
        assert!(row_len.iter().all(|v| v == &row_len[0]), "Jagged matrix provided to new_2d");

        let length = data.len();
        let shape = match row_len[0] {
            0 => vec![0, 0],
            _ => vec![data.len(), row_len[0]]
        };
        unsafe {
            let grad = if REQUIRES_GRAD {
                Some(RefCell::new(vec![T::zero(); length]))
            }
            else {
                None
            };

            Tensor {
                shape: shape,
                data: data.into_iter().flatten().collect(),
                grad: grad,
                requires_grad: REQUIRES_GRAD,
                back_prop_fn: None,
                back_prop_ctx: None,
            }
        }
    }
    
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Tensor<'a, T> {
        let length = data.len();
        assert_eq!(shape.iter().product::<usize>(), length);
        unsafe {
            let grad = if REQUIRES_GRAD {
                Some(RefCell::new(vec![T::zero(); length]))
            }
            else {
                None
            };
            Tensor {
                shape,
                data,
                grad: grad,
                requires_grad: REQUIRES_GRAD,
                back_prop_fn: None,
                back_prop_ctx: None,
            }
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Tensor<'a, T> {
        let data = vec![T::zero(); shape.iter().product()];
        Tensor::new(shape, data)
    }

    pub fn ones(shape: Vec<usize>) -> Tensor<'a, T> {
        let data = vec![T::one(); shape.iter().product()];
        Tensor::new(shape, data)
    }
}

// Misc
impl<'a, T: Dtype> Tensor<'a, T> {
    pub fn get_requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        if self.requires_grad == requires_grad {
            // Idempotent function; return early
            return;
        }

        self.requires_grad = requires_grad;
        if requires_grad {
            self.grad = Some(RefCell::new(vec![T::zero(); self.shape.iter().product()]));
        }
        else {
            // If the tensor doesn't need a grad, we can get rid of the one we had
            self.grad = None;
        }
    }

    fn check_shape(&self, other: &Tensor<T>) -> bool {
        // TODO Relax this constraint to allow broadcasting
        self.shape == other.shape
    }

    fn print_vec_as_tensor(shape: &Vec<usize>, vec: &Vec<T>, f: &mut Formatter<'_>) -> std::fmt::Result {
        match shape.len() {
            1 => { writeln!(f, "{:?}", vec) }
            2 => {
                let cols = shape[1];
                let rows = shape[0];
                write!(f, "Shape: {:?} ", shape)?;
                for i in 0..rows {
                    let row = &vec[i * cols..(i + 1) * cols];
                    write!(f, "{:?}", row)?;
                }
                writeln!(f, "")
            }
            // TODO: I'd like to handle this more nicely, by printing the tensor in 2D blocks, the way PyTorch does
            _ => { writeln!(f, "{:?}: {:?}", shape, vec) }
        }
    }
}
impl<'a, T: Dtype> Display for Tensor<'a, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Data:")?;
        Tensor::print_vec_as_tensor(&self.shape, &self.data, f)?;
        if let Some(grad) = &self.grad {
            writeln!(f, "Grad:")?;
            Tensor::print_vec_as_tensor(&self.shape, &grad.borrow(), f)?;
        }
        Ok(())
    }
}
impl<'a, T> PartialEq for Tensor<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}
impl<'a, T> Eq for Tensor<'a, T> {}
impl<'a, T> Hash for Tensor<'a, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::hash(self, state);
    }
}

// Autograd
impl<'a, T: Dtype> Tensor<'a, T> {
    fn update_grad(grad: &mut Vec<T>, updates: &Vec<T>) {
        for (g, u) in grad.iter_mut().zip(updates.iter()) {
            *g += *u;
        }
    }

    pub fn visit<'b>(
        topological_ordering: &mut Vec<&'b Tensor<'a, T>>,
        visited: &mut HashSet<&'b Tensor<'a, T>>,
        tensor: &'b Tensor<'a, T>) {
        if !visited.contains(tensor) {
            visited.insert(tensor);
            if let Some(lhs) = tensor.back_prop_ctx.as_ref().map(|ctx| ctx.rhs.unwrap()) {
                Tensor::visit(topological_ordering, visited, lhs);
            }
            if let Some(rhs) = tensor.back_prop_ctx.as_ref().map(|ctx| ctx.rhs.unwrap()) {
                Tensor::visit(topological_ordering, visited, rhs);
            }
            topological_ordering.push(&tensor);
        }
    }
    
    pub fn bwd(&self) -> Result<(), TensorError> {
        if !self.requires_grad {
            return Err(TensorError::NoGrad);
        }

        // Build a topological ordering of the graph
        let mut topological_ordering = Vec::<&Tensor<'a, T>>::new();
        let mut visited = HashSet::<&Tensor<'a, T>>::new();
        Tensor::visit(&mut topological_ordering, &mut visited, &self);

        self.grad.as_ref().as_mut().map(|ref_cell| ref_cell.borrow_mut().iter_mut().for_each(|x| *x = T::one()));
        for tensor in topological_ordering.iter().rev() {
            if let Some(ctx) = tensor.back_prop_ctx.as_ref() {
                tensor.back_prop_fn.expect("Backpropagation function was None")(tensor, &ctx);
            }
        }
        Ok(())
    }
}

// Operators
impl<'a, T: Dtype> Tensor<'a, T> {
    fn add_bwd(&self, ctx: &BackPropCtx<'a, T>) {
        let updates = self.grad.as_ref().expect("Self gradient was None during add back propagation.").borrow();
        let mut lhs_grad = ctx.lhs.expect("LHS Tensor was none during add back propagation")
            .grad.as_ref().expect("LHS gradient was None during back propagation.").borrow_mut();
        let mut rhs_grad = ctx.rhs.expect("RHS Tensor was none during add back propagation")
            .grad.as_ref().expect("RHS gradient was None during back propagation.").borrow_mut();

        Tensor::update_grad(&mut lhs_grad, &updates);
        Tensor::update_grad(&mut rhs_grad, &updates);
    }
}
impl<'a, T: Dtype> Add<&'a Tensor<'a, T>> for &'a Tensor<'a, T> {
    type Output = Tensor<'a, T>;
    fn add(self, rhs: &'a Tensor<'a, T>) -> Tensor<'a, T> {
        assert!(self.check_shape(rhs), "Cannot add tensors with different shapes ({:?} !+ {:?}",
            self.shape, rhs.shape);
        let res_data = self.data.iter().zip(rhs.data.iter()).map(|(&a, &b)| a + b).collect();
        let mut res = Tensor::new(self.shape.clone(), res_data);
        if self.requires_grad {
            res.back_prop_fn = Some(Tensor::add_bwd);
            res.back_prop_ctx = Some(BackPropCtx { lhs: Some(self), rhs: Some(rhs) });
        }
        else {
            res.requires_grad = false;
        }
        res
    }
}
// TODO I think there's probably a way to implement a non-borrowing overloads, but I'd only bother if I can leverage
// the borrowing versions; I don't want to duplicate code

#[derive(Debug, Default)]
struct BackPropCtx<'a, T> {
    lhs: Option<&'a Tensor<'a, T>>,
    rhs: Option<&'a Tensor<'a, T>>,
}

#[cfg(test)]
mod constructor_tests {
    use crate::Tensor;

    #[test]
    fn new_0d() {
        let actual = Tensor::new_0d(1i8);
        assert_eq!(actual.shape, vec![1]);
        assert_eq!(actual.data, vec![1i8]);

        let actual = Tensor::new_0d(0_i16);
        assert_eq!(actual.shape, vec![1]);
        assert_eq!(actual.data, vec![0_i16]);
        
        let actual = Tensor::new_0d(-3);
        assert_eq!(actual.shape, vec![1]);
        assert_eq!(actual.data, vec![-3]);

        let actual = Tensor::new_0d(1.0_f32);
        assert_eq!(actual.shape, vec![1]);
        assert_eq!(actual.data, vec![1.0_f32]);

        let actual = Tensor::new_0d(-1.5);
        assert_eq!(actual.shape, vec![1]);
        assert_eq!(actual.data, vec![-1.5]);
    }

    #[test]
    fn new_1d() {
        let empty: Tensor<f32> = Tensor::new_1d(vec![]);
        assert_eq!(empty.shape, vec![0]);
        assert_eq!(empty.data.len(), 0);

        let actual = Tensor::new_1d(vec![1.0_f32, 2.0_f32, 3.0_f32]);
        assert_eq!(actual.shape, vec![3]);
        assert_eq!(actual.data, vec![1.0_f32, 2.0_f32, 3.0_f32]);
    }

    #[test]
    fn new_2d() {
        let empty: Tensor<f32> = Tensor::new_2d(vec![vec![]]);
        assert_eq!(empty.shape, vec![0,0]);
        assert_eq!(empty.data.len(), 0);

        let actual = Tensor::new_2d(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0]]
        );
        assert_eq!(actual.shape, vec![2,3]);
        assert_eq!(actual.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
    #[test]
    #[should_panic(expected = "")]
    fn new_2d_jagged() {
        let _ = Tensor::new_2d(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0]]
        );
    }

    #[test]
    fn new() {
        let empty: Tensor<f64> = Tensor::new(vec![0,0,0], vec![]);
        assert_eq!(empty.shape, vec![0,0,0]);
        assert_eq!(empty.data.len(), 0);

        // 0D
        let actual = Tensor::new(vec![1], vec![1.0]);
        assert_eq!(actual.shape, vec![1]);
        assert_eq!(actual.data, vec![1.0]);

        // 1D
        let actual = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);
        assert_eq!(actual.shape, vec![3]);
        assert_eq!(actual.data, vec![1.0, 2.0, 3.0]);

        // 2D
        let actual = Tensor::new(vec![2,3], vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        ]);
        assert_eq!(actual.shape, vec![2,3]);
        assert_eq!(actual.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let actual = Tensor::new(vec![3,2], vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        ]);
        assert_eq!(actual.shape, vec![3,2]);
        assert_eq!(actual.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // 3D
        let actual = Tensor::new(vec![2,3,2], vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        ]);
        assert_eq!(actual.shape, vec![2,3,2]);
        assert_eq!(actual.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        // 4D
        let actual = Tensor::new(vec![2,2,2,2], vec![
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
        assert_eq!(actual.shape, vec![2,2,2,2]);
        assert_eq!(actual.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                     9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);

        // TODO: Add more--for example, what about shape: [1,0,1]?
    }
    #[test]
    #[should_panic(expected = "")]
    fn new_invalid_shape() {
        let _ = Tensor::new(vec![2], vec![1.0, 2.0, 3.0]);
    }
    
    #[test]
    fn zeros() {
        // 0D
        let d0: Tensor<f64> = Tensor::zeros(vec![1]);
        assert_eq!(d0.shape, vec![1]);
        assert_eq!(d0.data, vec![0.0; 1]);

        // 1D
        let d1: Tensor<f64> = Tensor::zeros(vec![5]);
        assert_eq!(d1.shape, vec![5]);
        assert_eq!(d1.data, vec![0.0; 5]);

        // 2D
        let d1: Tensor<f64> = Tensor::zeros(vec![15, 30]);
        assert_eq!(d1.shape, vec![15, 30]);
        assert_eq!(d1.data, vec![0.0; 15 * 30]);

        // 3D
        let d1: Tensor<f64> = Tensor::zeros(vec![15, 30, 100]);
        assert_eq!(d1.shape, vec![15, 30, 100]);
        assert_eq!(d1.data, vec![0.0; 15 * 30 * 100]);

        // 4D
        let d1: Tensor<f64> = Tensor::zeros(vec![15, 30, 100, 5]);
        assert_eq!(d1.shape, vec![15, 30, 100, 5]);
        assert_eq!(d1.data, vec![0.0; 15 * 30 * 100 * 5]);
    }
    
    #[test]
    fn ones() {
        // 0D
        let d0: Tensor<f64> = Tensor::ones(vec![1]);
        assert_eq!(d0.shape, vec![1]);
        assert_eq!(d0.data, vec![1.0; 1]);

        // 1D
        let d1: Tensor<f64> = Tensor::ones(vec![5]);
        assert_eq!(d1.shape, vec![5]);
        assert_eq!(d1.data, vec![1.0; 5]);

        // 2D
        let d1: Tensor<f64> = Tensor::ones(vec![15, 30]);
        assert_eq!(d1.shape, vec![15, 30]);
        assert_eq!(d1.data, vec![1.0; 15 * 30]);

        // 3D
        let d1: Tensor<f64> = Tensor::ones(vec![15, 30, 100]);
        assert_eq!(d1.shape, vec![15, 30, 100]);
        assert_eq!(d1.data, vec![1.0; 15 * 30 * 100]);

        // 4D
        let d1: Tensor<f64> = Tensor::ones(vec![15, 30, 100, 5]);
        assert_eq!(d1.shape, vec![15, 30, 100, 5]);
        assert_eq!(d1.data, vec![1.0; 15 * 30 * 100 * 5]);
    }
}

#[cfg(test)]
mod misc_tests {
    use crate::Tensor;

    #[test]
    fn grad_accessor() {
        // 0d
        let mut tensor = Tensor::new(vec![1], vec![5.0]);
        assert!(!tensor.get_requires_grad());
        assert!(tensor.grad.is_none());

        {
            tensor.set_requires_grad(true);
            let grad = tensor.grad.as_ref().unwrap().borrow_mut();
            assert_eq!(*grad, vec![0.0; tensor.shape.iter().product()]);
        }

        tensor.set_requires_grad(false);
        assert!(tensor.grad.is_none());
    }
}

#[cfg(test)]
mod operator_tests {
    use crate::error::TensorError;
    use crate::Tensor;
    use crate::tensor::Dtype;
    use crate::tensor::REQUIRES_GRAD;

    fn check_add<'a, T: Dtype>(lhs: &'a Tensor<'a, T>, rhs: &'a Tensor<'a, T>, expected: Vec<T>) {
        let actual = lhs + rhs;
        assert_eq!(actual.data, expected);

        actual.bwd().unwrap();
        let lhs_grad = lhs.grad.as_ref().unwrap().borrow();
        let rhs_grad = rhs.grad.as_ref().unwrap().borrow();
        assert_eq!(*lhs_grad, vec![T::one(); lhs.shape.iter().product()]);
        assert_eq!(*rhs_grad, vec![T::one(); rhs.shape.iter().product()]);
    }

    #[test]
    fn add_with_grad() {
        unsafe {REQUIRES_GRAD = true;}
        // 0d
        let a = Tensor::new(vec![1], vec![5.0_f32]);
        let b = Tensor::new(vec![1], vec![6.0_f32]);
        let expected = vec![11.0_f32];
        check_add(&a, &b, expected);

        // 1d
        let a = Tensor::new(vec![3], vec![1.0, -1.0, 2.0]);
        let b = Tensor::new(vec![3], vec![2.0, 3.0, -6.0]);
        let expected = vec![3.0, 2.0, -4.0];
        check_add(&a, &b, expected);

        // 2d
        let a = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let b = Tensor::new(vec![2, 2], vec![1.0, 4.0, 7.0, 10.0]);
        let expected = vec![6.0, 10.0, 14.0, 18.0];
        check_add(&a, &b, expected);

        // 3d
        let a = Tensor::new(vec![2, 2, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let b = Tensor::new(vec![2, 2, 2], vec![4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]);
        let expected = vec![4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0];
        check_add(&a, &b, expected);

        // 4d
        let a = Tensor::new(vec![2, 2, 2, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        let b = Tensor::new(vec![2, 2, 2, 2], vec![8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0]);
        let expected = vec![8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0, 35.0, 38.0, 41.0, 44.0, 47.0, 50.0, 53.0];
        check_add(&a, &b, expected);
    }
    #[test]
    fn add_no_grad() {
        // 0d
        let a = Tensor::new(vec![1], vec![5.0]);
        let b = Tensor::new(vec![1], vec![6.0]);
        let actual = &a + &b;
        let expected = vec![11.0];
        assert_eq!(actual.data, expected);
        assert!(a.grad.is_none());
        assert!(b.grad.is_none());
        assert_eq!(actual.bwd(), Err(TensorError::NoGrad));
    }
    #[test]
    fn add_int() {
        // 0d
        let a = Tensor::new(vec![1], vec![50]);
        let b = Tensor::new(vec![1], vec![60]);
        let actual = &a + &b;
        let expected = vec![110];
        assert_eq!(actual.data, expected);
        assert!(a.grad.is_none());
        assert!(b.grad.is_none());
        assert_eq!(actual.bwd(), Err(TensorError::NoGrad));
    }
    #[test]
    #[should_panic(expected = "")]
    fn add_shape_mismatch() {
        // 0d
        let a = Tensor::new(vec![1], vec![5.0]);
        let b = Tensor::new(vec![3], vec![2.0, 3.0, -6.0]);
        let expected = vec![7.0, 3.0, -6.0];
        check_add(&a, &b, expected);
    }
}