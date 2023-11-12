
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
    shape: Vec<usize>,
    data: Vec<T>,
    grad: Option<RefCell<Vec<T>>>,
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
                shape: vec![],
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

// Accessors
impl<'a, T: Dtype> Tensor<'a, T> {
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn get(&self, mut indices: Vec<usize>) -> Tensor<'a, T> {
        if indices.len() > self.rank() {
            panic!("Cannot index into a rank-{} tensor with {:?}. For a rank-0 tensor, call item() instead",
                self.rank(), indices);
        }

        let num_indices = indices.len();

        // The start position is the indices vec, extended to the length of the shape vector with 0s
        indices.resize(self.rank(), 0);
        let first_pos = self.array_pos(&indices);

        // The last position is the indices vec with the new zeros replaced by each dimension's respective maximum
        for i in num_indices..self.rank() {
            indices[i] = self.shape[i] - 1;
        }

        let last_pos = self.array_pos(&indices);

        // TODO Copying here has grave performance implications for very large tensors, as are quite common in machine
        // learning applications. Some better approaches:
        //  - Wrap data in a RefCell<> and maybe an RC<>, and have the tensor returned here refer to the same underlying
        //  memory
        //  - Rather something other than a Tensor here (sort of like String and str)
        Tensor::new(self.shape[num_indices..self.rank()].to_vec(), self.data[first_pos..last_pos+1].to_vec())
    }

    pub fn item(&self) -> T {
        if self.rank() == 0 || self.shape.iter().product::<usize>() == 1 {
            self.data[0]
        }
        else {
            panic!("Cannot call item() on a rank-{} tensor", self.rank());
        }

    }

    pub fn element(&self, indices: Vec<usize>) -> T {
        let pos = self.array_pos(&indices);
        self.data[pos]
    }

    pub fn set_element(&mut self, indices: Vec<usize>, value: T) {
        let pos = self.array_pos(&indices);
        self.data[pos] = value;
    }

    pub fn increment_element(&mut self, indices: Vec<usize>) {
        let element = self.element(indices.clone());
        self.set_element(indices, element + T::one());
    }

    pub fn requires_grad(&self) -> bool {
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

    // Given the coordinates of an individual element, return its position within the backing array.
    fn array_pos(&self, indices: &Vec<usize>) -> usize {
        if indices.len() != self.rank() {
            panic!("Cannot index into a rank-{} tensor with {:?}.", self.rank(), indices);
        }

        for i in 0..indices.len() {
            if indices[i] >= self.shape[i] {
                panic!("Index {} ({}) is out of bounds for shape {:?}", i, indices[i], self.shape);
            }
        }

        indices.iter().zip(self.shape.iter()).fold(0, |acc, (&index, &dim)| acc * dim + index)
    }
}

// Misc
impl<'a, T: Dtype> Tensor<'a, T> {
    fn check_shape(&self, other: &Tensor<T>) -> bool {
        // TODO Relax this constraint to allow broadcasting
        self.shape == other.shape
    }

    fn print_vec_as_tensor(shape: &Vec<usize>, vec: &Vec<T>, f: &mut Formatter<'_>) -> std::fmt::Result {
        match shape.len() {
            1 => { writeln!(f, " {:?}", vec) }
            2 => {
                let rows = shape[0];
                let cols = shape[1];
                let first_row = &vec[0..cols];
                write!(f, "\n[{:?}", first_row)?;
                if rows == 1 {
                    writeln!(f, "]")?;
                    return Ok(());
                }
                else {
                    writeln!(f, "")?;
                }
                for i in 1..rows-1 {
                    let row = &vec[(i * cols)..((i + 1) * cols)];
                    writeln!(f, " {:?}", row)?;
                }
                writeln!(f, " {:?}]", &vec[((rows-1) * cols)..((rows) * cols)])
            }
            // TODO: I'd like to handle this more nicely, by printing the tensor in 2D blocks, the way PyTorch does
            _ => { writeln!(f, " {:?}: {:?}", shape, vec) }
        }
    }
}
impl<'a, T: Dtype> Display for Tensor<'a, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Data:")?;
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

// Automatic differentiation
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
// TODO I think there's probably a way to implement a non-borrowing operator overloads, but I'd only bother if I can
// leverage the borrowing versions; I don't want to duplicate code

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
        assert_eq!(actual.shape, vec![]);
        assert_eq!(actual.data, vec![1i8]);
        assert_eq!(actual.item(), 1i8);

        let actual = Tensor::new_0d(0i16);
        assert_eq!(actual.shape, vec![]);
        assert_eq!(actual.data, vec![0i16]);
        assert_eq!(actual.item(), 0i16);
        
        let actual = Tensor::new_0d(-3);
        assert_eq!(actual.shape, vec![]);
        assert_eq!(actual.data, vec![-3]);

        let actual = Tensor::new_0d(1.0f32);
        assert_eq!(actual.shape, vec![]);
        assert_eq!(actual.data, vec![1.0f32]);

        let actual = Tensor::new_0d(-1.5);
        assert_eq!(actual.shape, vec![]);
        assert_eq!(actual.data, vec![-1.5]);
    }

    #[test]
    fn new_1d() {
        let empty: Tensor<f32> = Tensor::new_1d(vec![]);
        assert_eq!(empty.shape, vec![0]);
        assert_eq!(empty.data.len(), 0);

        let actual = Tensor::new_1d(vec![1.0f32, 2.0f32, 3.0f32]);
        assert_eq!(actual.shape, vec![3]);
        assert_eq!(actual.data, vec![1.0f32, 2.0f32, 3.0f32]);
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
        let d2: Tensor<f64> = Tensor::ones(vec![15, 30]);
        assert_eq!(d2.shape, vec![15, 30]);
        assert_eq!(d2.data, vec![1.0; 15 * 30]);

        // 3D
        let d3: Tensor<f64> = Tensor::ones(vec![15, 30, 100]);
        assert_eq!(d3.shape, vec![15, 30, 100]);
        assert_eq!(d3.data, vec![1.0; 15 * 30 * 100]);

        // 4D
        let d4: Tensor<f64> = Tensor::ones(vec![15, 30, 100, 5]);
        assert_eq!(d4.shape, vec![15, 30, 100, 5]);
        assert_eq!(d4.data, vec![1.0; 15 * 30 * 100 * 5]);
    }
}

#[cfg(test)]
mod accessor_tests {
    use crate::Tensor;

    // TODO Rank tests

    #[test]
    fn get() {
        let d4 = Tensor::new(vec![2,3,4,5], (0..120).map(|i| i as i32).collect());

        // Naming assumes NCWH
        for n in 0..d4.shape[0] {
            let d3 = d4.get(vec![n]);
            for c in 0..d4.shape[1] {
                let d2 = d3.get(vec![c]);
                for h in 0..d4.shape[2] {
                    let d1 = d2.get(vec![h]);
                    for w in 0..d4.shape[3] {
                        let d0 = d1.get(vec![w]);
                        let expected = n * d4.shape.iter().skip(1).product::<usize>() +
                            c * d4.shape.iter().skip(2).product::<usize>() +
                            h * d4.shape.iter().skip(3).product::<usize>() +
                            w;
                        assert_eq!(d0.item() as usize, expected);
                        assert_eq!(d4.get(vec![n, c, h, w]).item() as usize, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn grad_accessor() {
        // 0d
        let mut tensor = Tensor::new(vec![1], vec![5.0]);
        assert!(!tensor.requires_grad());
        assert!(tensor.grad.is_none());

        {
            tensor.set_requires_grad(true);
            let grad = tensor.grad.as_ref().unwrap().borrow_mut();
            assert_eq!(*grad, vec![0.0; tensor.shape.iter().product()]);
        }

        tensor.set_requires_grad(false);
        assert!(tensor.grad.is_none());
    }

    #[test]
    fn element_accessor() {
        let mut d0 = Tensor::new_0d(1);
        let mut d1 = Tensor::new_1d((1..11).collect());
        let mut d2 = Tensor::new(vec![3,3],(0..9).map(|x| x as f32).collect());
        let mut d3 = Tensor::new(vec![3,3,3],(0..27).map(|x| x as f32).collect());
        let mut d4 = Tensor::new(vec![3,3,3,3],(0..81).map(|x| x as f32).collect());

        assert_eq!(d0.element(vec![]), 1);
        assert_eq!(d1.element(vec![1]), 2);
        assert_eq!(d1.element(vec![9]), 10);
        assert_eq!(d2.element(vec![0,1]), 1.0);
        assert_eq!(d2.element(vec![1,1]), 4.0);
        assert_eq!(d2.element(vec![2,2]), 8.0);
        assert_eq!(d3.element(vec![0,0,2]), 2.0);
        assert_eq!(d3.element(vec![1,1,1]), 13.0);
        assert_eq!(d3.element(vec![2,0,0]), 18.0);
        assert_eq!(d4.element(vec![0,0,2,1]), 7.0);
        assert_eq!(d4.element(vec![1,1,1,0]), 39.0);
        assert_eq!(d4.element(vec![2,0,0,2]), 56.0);

        d0.set_element(vec![], 0);
        d1.set_element(vec![1], 0);
        d1.set_element(vec![9], 0);
        d2.set_element(vec![0,1], 0.0);
        d2.set_element(vec![1,1], 0.0);
        d2.set_element(vec![2,2], 0.0);
        d3.set_element(vec![0,0,2], 0.0);
        d3.set_element(vec![1,1,1], 0.0);
        d3.set_element(vec![2,0,0], 0.0);
        d4.set_element(vec![0,0,2,1], 0.0);
        d4.set_element(vec![1,1,1,0], 0.0);
        d4.set_element(vec![2,0,0,2], 0.0);

        assert_eq!(d0.element(vec![]), 0);
        assert_eq!(d1.element(vec![1]), 0);
        assert_eq!(d1.element(vec![9]), 0);
        assert_eq!(d2.element(vec![0,1]), 0.0);
        assert_eq!(d2.element(vec![1,1]), 0.0);
        assert_eq!(d2.element(vec![2,2]), 0.0);
        assert_eq!(d3.element(vec![0,0,2]), 0.0);
        assert_eq!(d3.element(vec![1,1,1]), 0.0);
        assert_eq!(d3.element(vec![2,0,0]), 0.0);
        assert_eq!(d4.element(vec![0,0,2,1]), 0.0);
        assert_eq!(d4.element(vec![1,1,1,0]), 0.0);
        assert_eq!(d4.element(vec![2,0,0,2]), 0.0);

        d0.increment_element(vec![]);
        d1.increment_element(vec![1]);
        d1.increment_element(vec![9]);
        d2.increment_element(vec![0,1]);
        d2.increment_element(vec![1,1]);
        d2.increment_element(vec![2,2]);
        d3.increment_element(vec![0,0,2]);
        d3.increment_element(vec![1,1,1]);
        d3.increment_element(vec![2,0,0]);
        d4.increment_element(vec![0,0,2,1]);
        d4.increment_element(vec![1,1,1,0]);
        d4.increment_element(vec![2,0,0,2]);

        assert_eq!(d0.element(vec![]), 1);
        assert_eq!(d1.element(vec![1]), 1);
        assert_eq!(d1.element(vec![9]), 1);
        assert_eq!(d2.element(vec![0,1]), 1.0);
        assert_eq!(d2.element(vec![1,1]), 1.0);
        assert_eq!(d2.element(vec![2,2]), 1.0);
        assert_eq!(d3.element(vec![0,0,2]), 1.0);
        assert_eq!(d3.element(vec![1,1,1]), 1.0);
        assert_eq!(d3.element(vec![2,0,0]), 1.0);
        assert_eq!(d4.element(vec![0,0,2,1]), 1.0);
        assert_eq!(d4.element(vec![1,1,1,0]), 1.0);
        assert_eq!(d4.element(vec![2,0,0,2]), 1.0);
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
        let a = Tensor::new(vec![1], vec![5.0f32]);
        let b = Tensor::new(vec![1], vec![6.0f32]);
        let expected = vec![11.0f32];
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