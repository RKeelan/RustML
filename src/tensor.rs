
use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div};

use derivative::Derivative;

use crate::Dtype;
use crate::TensorError;


// TODO: Following PyTorch, disallow gradients for integer-based tensors

#[derive(Derivative)]
#[derivative(Debug)]
pub struct Tensor<'a, T> {
    shape: Vec<usize>,
    data: Vec<T>,
    grad: Option<RefCell<Vec<T>>>,
    #[derivative(Debug="ignore")]
    back_prop_fn: Option<fn(result: &Tensor<'a, T>, ctx: &BackPropCtx<'a, T>)>,
    back_prop_ctx: Option<BackPropCtx<'a, T>>,
}

// Constructors
impl<'a, T: Dtype> Tensor<'a, T> {
    // Without gradient
    pub fn new_0d(data: T, requires_grad: bool) -> Self {
        let grad = if requires_grad {
            Some(RefCell::new(vec![T::zero(); 1]))
        }
        else {
            None
        };
        Tensor {
            shape: vec![],
            data: vec![data],
            grad: grad,
            back_prop_fn: None,
            back_prop_ctx: None,
        }
    }

    pub fn new_1d(data: Vec<T>, requires_grad: bool) -> Self {
        let length = data.len();
        let grad = if requires_grad {
            Some(RefCell::new(vec![T::zero(); length]))
        }
        else {
            None
        };
        Tensor {
            shape: vec![data.len()],
            data,
            grad: grad,
            back_prop_fn: None,
            back_prop_ctx: None,
        }
    }

    pub fn new_2d(data: Vec<Vec<T>>, requires_grad: bool) -> Self {
        // Check for jaggedness
        let row_len = data.iter().map(|v| v.len()).collect::<Vec<usize>>();
        assert!(row_len.iter().all(|v| v == &row_len[0]), "Jagged matrix provided to new_2d");

        let shape = match row_len[0] {
            0 => vec![0, 0],
            _ => vec![data.len(), row_len[0]]
        };
        let data: Vec<T> = data.into_iter().flatten().collect();
        let grad = if requires_grad {
            Some(RefCell::new(vec![T::zero(); data.len()]))
        }
        else {
            None
        };
        Tensor {
            shape: shape,
            data: data,
            grad: grad,
            back_prop_fn: None,
            back_prop_ctx: None,
        }
    }
    
    pub fn new(shape: Vec<usize>, data: Vec<T>, requires_grad: bool) -> Self {
        let length = data.len();
        assert_eq!(shape.iter().product::<usize>(), length);
        let grad = if requires_grad {
            Some(RefCell::new(vec![T::zero(); length]))
        }
        else {
            None
        };
        Tensor {
            shape,
            data,
            grad: grad,
            back_prop_fn: None,
            back_prop_ctx: None,
        }
    }

    pub fn zeros(shape: Vec<usize>, requires_grad: bool) -> Self {
        let data = vec![T::zero(); shape.iter().product()];
        Tensor::new(shape, data, requires_grad)
    }

    pub fn ones(shape: Vec<usize>, requires_grad: bool) -> Self {
        let data = vec![T::one(); shape.iter().product()];
        Tensor::new(shape, data,requires_grad)
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

    pub fn get(&self, mut indices: Vec<usize>) -> Self {
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
        Tensor::new(self.shape[num_indices..self.rank()].to_vec(), self.data[first_pos..last_pos+1].to_vec(),
            self.requires_grad())
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
        self.grad != None
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        if self.requires_grad() == requires_grad {
            // Idempotent function; return early
            return;
        }

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
        // TODO Also print the DType
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
            if let Some(ctx) = tensor.back_prop_ctx.as_ref() {
                if let Some(lhs) = ctx.rhs {
                    Tensor::visit(topological_ordering, visited, lhs);
                }
            }
            if let Some(ctx) = tensor.back_prop_ctx.as_ref(){
                if let Some(rhs) = ctx.rhs {
                    Tensor::visit(topological_ordering, visited, rhs);
                }
            }
            topological_ordering.push(&tensor);
        }
    }
    
    pub fn bwd(&self) -> Result<(), TensorError> {
        if self.grad == None {
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
// TODO I think there's probably a way to implement a non-borrowing operator overloads, but I'd only bother if I can
// leverage the borrowing versions; I don't want to duplicate code
impl<'a, T: Dtype> Tensor<'a, T> {
    fn add_bwd(&self, ctx: &BackPropCtx<'a, T>) {
        let updates = self.grad.as_ref().expect("Self gradient was None during add back propagation.").borrow();
        let mut lhs_grad = ctx.lhs.expect("LHS Tensor was none during add back propagation")
            .grad.as_ref().expect("LHS gradient was None during add back propagation.").borrow_mut();
        let mut rhs_grad = ctx.rhs.expect("RHS Tensor was none during add back propagation")
            .grad.as_ref().expect("RHS gradient was None during add back propagation.").borrow_mut();

        Tensor::update_grad(&mut lhs_grad, &updates);
        Tensor::update_grad(&mut rhs_grad, &updates);
    }
}
impl<'a, T: Dtype> Add<&'a Tensor<'a, T>> for &'a Tensor<'a, T> {
    type Output = Tensor<'a, T>;
    fn add(self, rhs: &'a Tensor<'a, T>) -> Tensor<'a, T> {
        assert!(self.is_broadcastable(rhs), "Cannot add tensors with different shapes ({:?} !+ {:?}",
            self.shape, rhs.shape);
        let res_data = self.data.iter().zip(rhs.data.iter()).map(|(&a, &b)| a + b).collect();
        let requires_grad = self.requires_grad() || rhs.requires_grad();
        let mut res = Tensor::new(self.shape.clone(), res_data, requires_grad);
        if requires_grad {
            res.back_prop_fn = Some(Tensor::add_bwd);
            res.back_prop_ctx = Some(BackPropCtx { lhs: Some(self), rhs: Some(rhs) });
        }
        res
    }
}
impl<'a, T: Dtype> Tensor<'a, T> {
    fn div_bwd(&self, ctx: &BackPropCtx<'a, T>) {
        let self_grad = self.grad.as_ref().expect("Self gradient was None during dev back propagation.").borrow();
        // Given y - LHS / RHS; LHS is the numerator and RHS is the denominator
        let numerator_data = &ctx.lhs.expect("LHS (numerator) Tensor was none during dev back propagation").data;
        let denominator = &ctx.rhs.expect("RHS (denominator) Tensor was none during dev back propagation").data;

        let two = T::one() + T::one();
        let numerator_updates = denominator.iter().zip(self_grad.iter()).map(|(&r, &g)| g / r).collect();
        let denominator_updates = numerator_data.iter().zip(denominator.iter()).zip(self_grad.iter())
            .map(|((&l, &r), &g)| {
                let result = -(l / r.pow(two)) * g;
                result
        }).collect();
        
        let mut numerator_grad = ctx.lhs.expect("LHS (numerator) Tensor was none during dev back propagation")
            .grad.as_ref().expect("LHS (numerator) gradient was None during add back propagation.").borrow_mut();
        let mut denominator_grad = ctx.rhs.expect("RHS (denominator) Tensor was none during dev back propagation")
            .grad.as_ref().expect("RHS (denominator) gradient was None during add back propagation.").borrow_mut();

        Tensor::update_grad(&mut numerator_grad, &numerator_updates);
        Tensor::update_grad(&mut denominator_grad, &denominator_updates);
    }
}
impl<'a, T: Dtype> Div<&'a Tensor<'a, T>> for &'a Tensor<'a, T> {
    type Output = Tensor<'a, T>;
    fn div(self, rhs: &'a Tensor<'a, T>) -> Tensor<'a, T> {
        assert!(self.is_broadcastable(rhs), "Cannot divide tensors with different shapes ({:?} !+ {:?}",
            self.shape, rhs.shape);
        let res_data = self.data.iter().zip(rhs.data.iter()).map(|(&a, &b)| a / b).collect();
        let requires_grad = self.requires_grad() || rhs.requires_grad();
        let mut res = Tensor::new(self.shape.clone(), res_data, requires_grad);
        if requires_grad {
            res.back_prop_fn = Some(Tensor::div_bwd);
            res.back_prop_ctx = Some(BackPropCtx { lhs: Some(self), rhs: Some(rhs) });
        }
        res
    }
}

// Functions
impl<'a, T: Dtype> Tensor<'a, T> {
    fn sum_bwd(&self, ctx: &BackPropCtx<'a, T>) {
        let updates = self.grad.as_ref().expect("Self gradient was None during sum back propagation.").borrow();
        assert!(updates.len() == 1, "Sum() should result in a scalar tensor");
        let mut lhs_grad = ctx.lhs.expect("LHS Tensor was none during sum back propagation")
            .grad.as_ref().expect("LHS gradient was None during sum back propagation.").borrow_mut();
        let updates = vec![updates[0]; lhs_grad.len()];

        Tensor::update_grad(&mut lhs_grad, &updates);
    }
    pub fn sum(&'a self) -> Self {
        let res_data: T = self.data.iter().map(|x| *x).sum();
        let requires_grad = self.grad != None;
        let mut res = Tensor::new_0d(res_data, requires_grad);
        if requires_grad {
            res.back_prop_fn = Some(Tensor::sum_bwd);
            res.back_prop_ctx = Some(BackPropCtx { lhs: Some(&self), rhs: None });
        }
        res
    }
}

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
        let actual = Tensor::new_0d(1i8, false);
        assert_eq!(actual.shape, vec![]);
        assert_eq!(actual.data, vec![1i8]);
        assert_eq!(actual.item(), 1i8);
        assert!(actual.grad.is_none());

        let actual = Tensor::new_0d(0i16, false);
        assert_eq!(actual.shape, vec![]);
        assert_eq!(actual.data, vec![0i16]);
        assert_eq!(actual.item(), 0i16);
        assert!(actual.grad.is_none());
        
        let actual = Tensor::new_0d(-3, true);
        assert_eq!(actual.shape, vec![]);
        assert_eq!(actual.data, vec![-3]);
        assert_eq!(actual.data.len(), actual.grad.unwrap().borrow().len());

        let actual = Tensor::new_0d(1.0f32, true);
        assert_eq!(actual.shape, vec![]);
        assert_eq!(actual.data, vec![1.0f32]);
        assert_eq!(actual.data.len(), actual.grad.unwrap().borrow().len());

        let actual = Tensor::new_0d(-1.5, true);
        assert_eq!(actual.shape, vec![]);
        assert_eq!(actual.data, vec![-1.5]);
        assert_eq!(actual.data.len(), actual.grad.unwrap().borrow().len());
    }

    #[test]
    fn new_1d() {
        let empty: Tensor<f32> = Tensor::new_1d(vec![], false);
        assert_eq!(empty.shape, vec![0]);
        assert_eq!(empty.data.len(), 0);
        assert!(empty.grad.is_none());

        let actual = Tensor::new_1d(vec![1.0f32, 2.0f32, 3.0f32], true);
        assert_eq!(actual.shape, vec![3]);
        assert_eq!(actual.data, vec![1.0f32, 2.0f32, 3.0f32]);
        assert_eq!(actual.data.len(), actual.grad.unwrap().borrow().len());
    }

    #[test]
    fn new_2d() {
        let empty: Tensor<f32> = Tensor::new_2d(vec![vec![]], false);
        assert_eq!(empty.shape, vec![0,0]);
        assert_eq!(empty.data.len(), 0);
        assert!(empty.grad.is_none());

        let actual = Tensor::new_2d(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0]],
            true
        );
        assert_eq!(actual.shape, vec![2,3]);
        assert_eq!(actual.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(actual.data.len(), actual.grad.unwrap().borrow().len());
    }
    #[test]
    #[should_panic(expected = "")]
    fn new_2d_jagged() {
        let _ = Tensor::new_2d(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0]],
            true
        );
    }

    #[test]
    fn new() {
        let empty: Tensor<f64> = Tensor::new(vec![0,0,0], vec![], false);
        assert_eq!(empty.shape, vec![0,0,0]);
        assert_eq!(empty.data.len(), 0);
        assert!(empty.grad.is_none());

        // 0D
        let actual = Tensor::new(vec![1], vec![1.0], false);
        assert_eq!(actual.shape, vec![1]);
        assert_eq!(actual.data, vec![1.0]);
        assert!(actual.grad.is_none());

        // 1D
        let actual = Tensor::new(vec![3], vec![1.0, 2.0, 3.0], false);
        assert_eq!(actual.shape, vec![3]);
        assert_eq!(actual.data, vec![1.0, 2.0, 3.0]);
        assert!(actual.grad.is_none());

        // 2D
        let actual = Tensor::new(vec![2,3], vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        ], true);
        assert_eq!(actual.shape, vec![2,3]);
        assert_eq!(actual.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(actual.data.len(), actual.grad.unwrap().borrow().len());

        let actual = Tensor::new(vec![3,2], vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        ], true);
        assert_eq!(actual.shape, vec![3,2]);
        assert_eq!(actual.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(actual.data.len(), actual.grad.unwrap().borrow().len());

        // 3D
        let actual = Tensor::new(vec![2,3,2], vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        ], true);
        assert_eq!(actual.shape, vec![2,3,2]);
        assert_eq!(actual.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(actual.data.len(), actual.grad.unwrap().borrow().len());
        
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
        ], true);
        assert_eq!(actual.shape, vec![2,2,2,2]);
        assert_eq!(actual.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                     9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
        assert_eq!(actual.data.len(), actual.grad.unwrap().borrow().len());

        // TODO: Add more--for example, what about shape: [1,0,1]?
    }
    #[test]
    #[should_panic(expected = "")]
    fn new_invalid_shape() {
        let _ = Tensor::new(vec![2], vec![1.0, 2.0, 3.0], false);
    }
    
    #[test]
    fn zeros() {
        // 0D
        let d0: Tensor<f64> = Tensor::zeros(vec![1], false);
        assert_eq!(d0.shape, vec![1]);
        assert_eq!(d0.data, vec![0.0; 1]);

        // 1D
        let d1: Tensor<f64> = Tensor::zeros(vec![5], false);
        assert_eq!(d1.shape, vec![5]);
        assert_eq!(d1.data, vec![0.0; 5]);

        // 2D
        let d1: Tensor<f64> = Tensor::zeros(vec![15, 30], false);
        assert_eq!(d1.shape, vec![15, 30]);
        assert_eq!(d1.data, vec![0.0; 15 * 30]);

        // 3D
        let d1: Tensor<f64> = Tensor::zeros(vec![15, 30, 100], false);
        assert_eq!(d1.shape, vec![15, 30, 100]);
        assert_eq!(d1.data, vec![0.0; 15 * 30 * 100]);

        // 4D
        let d1: Tensor<f64> = Tensor::zeros(vec![15, 30, 100, 5], false);
        assert_eq!(d1.shape, vec![15, 30, 100, 5]);
        assert_eq!(d1.data, vec![0.0; 15 * 30 * 100 * 5]);
    }
    
    #[test]
    fn ones() {
        // 0D
        let d0: Tensor<f64> = Tensor::ones(vec![1], false);
        assert_eq!(d0.shape, vec![1]);
        assert_eq!(d0.data, vec![1.0; 1]);

        // 1D
        let d1: Tensor<f64> = Tensor::ones(vec![5], false);
        assert_eq!(d1.shape, vec![5]);
        assert_eq!(d1.data, vec![1.0; 5]);

        // 2D
        let d2: Tensor<f64> = Tensor::ones(vec![15, 30], false);
        assert_eq!(d2.shape, vec![15, 30]);
        assert_eq!(d2.data, vec![1.0; 15 * 30]);

        // 3D
        let d3: Tensor<f64> = Tensor::ones(vec![15, 30, 100], false);
        assert_eq!(d3.shape, vec![15, 30, 100]);
        assert_eq!(d3.data, vec![1.0; 15 * 30 * 100]);

        // 4D
        let d4: Tensor<f64> = Tensor::ones(vec![15, 30, 100, 5], false);
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
        let d4 = Tensor::new(vec![2,3,4,5], (0..120).map(|i| i as i32).collect(), false);

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
        let mut tensor = Tensor::new(vec![1], vec![5.0], false);
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
        let mut d0 = Tensor::new_0d(1, false);
        let mut d1 = Tensor::new_1d((1..11).collect(), false);
        let mut d2 = Tensor::new(vec![3,3],(0..9).map(|x| x as f32).collect(), false);
        let mut d3 = Tensor::new(vec![3,3,3],(0..27).map(|x| x as f32).collect(), false);
        let mut d4 = Tensor::new(vec![3,3,3,3],(0..81).map(|x| x as f32).collect(), false);

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
mod add_tests {
    use crate::error::TensorError;
    use crate::Tensor;
    use crate::tensor::Dtype;

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
        // 0d
        let a = Tensor::new(vec![1], vec![5.0f32], true);
        let b = Tensor::new(vec![1], vec![6.0f32], true);
        let expected = vec![11.0f32];
        check_add(&a, &b, expected);

        // 1d
        let a = Tensor::new(vec![3], vec![1.0, -1.0, 2.0], true);
        let b = Tensor::new(vec![3], vec![2.0, 3.0, -6.0], true);
        let expected = vec![3.0, 2.0, -4.0];
        check_add(&a, &b, expected);

        // 2d
        let a = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0], true);
        let b = Tensor::new(vec![2, 2], vec![1.0, 4.0, 7.0, 10.0], true);
        let expected = vec![6.0, 10.0, 14.0, 18.0];
        check_add(&a, &b, expected);

        // 3d
        let a = Tensor::new(vec![2, 2, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], true);
        let b = Tensor::new(vec![2, 2, 2], vec![4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0], true);
        let expected = vec![4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0];
        check_add(&a, &b, expected);

        // 4d
        let a = Tensor::new(vec![2, 2, 2, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], true);
        let b = Tensor::new(vec![2, 2, 2, 2], vec![8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0], true);
        let expected = vec![8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0, 35.0, 38.0, 41.0, 44.0, 47.0, 50.0, 53.0];
        check_add(&a, &b, expected);
    }
    #[test]
    fn add_no_grad() {
        let a = Tensor::new(vec![1], vec![5.0], false);
        let b = Tensor::new(vec![1], vec![6.0], false);
        let actual = &a + &b;
        let expected = vec![11.0];
        assert_eq!(actual.data, expected);
        assert!(a.grad.is_none());
        assert!(b.grad.is_none());
        assert_eq!(actual.bwd(), Err(TensorError::NoGrad));
    }
    #[test]
    fn add_int() {
        let a = Tensor::new(vec![1], vec![50], false);
        let b = Tensor::new(vec![1], vec![60], false);
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
        let a = Tensor::new(vec![3], vec![5.0, 4.0], false);
        let b = Tensor::new(vec![3], vec![2.0, 3.0, -6.0], false);
        let _ = &a + &b;
    }
}

#[cfg(test)]
mod div_tests {
    use crate::error::TensorError;
    use crate::Tensor;
    use crate::tensor::Dtype;

    fn assert_almost_equal_vec<T: Dtype>(actual: &Vec<T>, expected: &Vec<T>) {
        actual.iter().zip(expected.iter()).for_each(|(a, e)| {
            if !T::almost_equal(a, e) {
                panic!("actual: {:?}, expected: {:?}, epsilon: {:?}, diff: {:?}, expected*epsilon: {:?}",
                    a, e, T::epsilon(), (*a - *e).abs(), T::epsilon() * *e);
            }
        });
    }

    fn check_div<'a, T: Dtype>(
        lhs: &'a Tensor<'a, T>,
        rhs: &'a Tensor<'a, T>,
        expected_data: Vec<T>,
        lhs_expected_grad: Option<Vec<T>>,
        rhs_expected_grad: Option<Vec<T>>) {
        let actual = lhs / rhs;
        assert_almost_equal_vec(&actual.data, &expected_data);

        actual.bwd().unwrap();
        let lhs_grad = lhs.grad.as_ref().map(|refcell_vec| refcell_vec.borrow().clone());
        let rhs_grad = rhs.grad.as_ref().map(|refcell_vec| refcell_vec.borrow().clone());
        assert_almost_equal_vec(&lhs_grad.unwrap(), &lhs_expected_grad.unwrap());
        assert_almost_equal_vec(&rhs_grad.unwrap(), &rhs_expected_grad.unwrap());
    }

    #[test]
    fn div_with_grad() {
        // 0d
        let a = Tensor::new(vec![1], vec![5.0], true);
        let b = Tensor::new(vec![1], vec![6.0], true);
        let expected_data = vec![0.8333333333333334];
        let lhs_expected_grad = Some(vec![0.16666666666666666]);
        let rhs_expected_grad = Some(vec![-0.1388888888888889]);
        check_div(&a, &b, expected_data, lhs_expected_grad, rhs_expected_grad);

        // 1d
        let a = Tensor::new(vec![3], vec![1.0, -1.0, 2.0], true);
        let b = Tensor::new(vec![3], vec![2.0, 3.0, -6.0], true);
        let expected_data = vec![0.5, -0.3333333333333333, -0.3333333333333333];
        let lhs_expected_grad = Some(vec![0.5, 0.3333333333333333, -0.16666666666666666]);
        let rhs_expected_grad = Some(vec![-0.25, 0.1111111111111111, -0.05555555555555555]);
        check_div(&a, &b, expected_data, lhs_expected_grad, rhs_expected_grad);

        // 2d
        let a = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0], true);
        let b = Tensor::new(vec![2, 2], vec![1.0, 4.0, 7.0, 10.0], true);
        let expected_data = vec![5.0, 1.5, 1.0, 0.8];
        let lhs_expected_grad = Some(vec![1.0, 0.25, 0.14285714285714285, 0.1]);
        let rhs_expected_grad = Some(vec![-5.0, -0.375, -0.14285714285714285, -0.08]);
        check_div(&a, &b, expected_data, lhs_expected_grad, rhs_expected_grad);

        // 3d
        let a = Tensor::new(vec![2, 2, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], true);
        let b = Tensor::new(vec![2, 2, 2], vec![4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0], true);
        let expected_data = vec![0.0, 0.16666666666666666, 0.25, 0.3, 0.3333333333333333, 0.35714285714285715, 0.375, 0.3888888888888889];
        let lhs_expected_grad = Some(vec![0.25, 0.16666666666666666, 0.125, 0.1, 0.08333333333333333, 0.07142857142857142, 0.0625, 0.05555555555555555]);
        let rhs_expected_grad = Some(vec![-0.0, -0.027777777777777776, -0.03125, -0.03, -0.027777777777777776, -0.025510204081632654, -0.0234375, -0.021604938271604937]);
        check_div(&a, &b, expected_data, lhs_expected_grad, rhs_expected_grad);

        // 4d
        let a = Tensor::new(vec![2, 2, 2, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], true);
        let b = Tensor::new(vec![2, 2, 2, 2], vec![8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0], true);
        let expected_data = vec![0.0, 0.1, 0.16666666666666666, 0.21428571428571427, 0.25, 0.2777777777777778, 0.3, 0.3181818181818182, 0.3333333333333333, 0.34615384615384615, 0.35714285714285715, 0.36666666666666664, 0.375, 0.38235294117647056, 0.3888888888888889, 0.39473684210526316];
        let lhs_expected_grad = Some(vec![0.125, 0.1, 0.08333333333333333, 0.07142857142857142, 0.0625, 0.05555555555555555, 0.05, 0.045454545454545456, 0.041666666666666664, 0.038461538461538464, 0.03571428571428571, 0.03333333333333333, 0.03125, 0.029411764705882353, 0.027777777777777776, 0.02631578947368421]);
        let rhs_expected_grad = Some(vec![-0.0, -0.01, -0.013888888888888888, -0.015306122448979591, -0.015625, -0.0154320987654321, -0.015, -0.014462809917355372, -0.013888888888888888, -0.013313609467455622, -0.012755102040816327, -0.012222222222222221, -0.01171875, -0.011245674740484428, -0.010802469135802469, -0.01038781163434903]);
        check_div(&a, &b, expected_data, lhs_expected_grad, rhs_expected_grad);
    }
    #[test]
    fn div_no_grad() {
        // 0d
        let a = Tensor::new(vec![1], vec![5.0], false);
        let b = Tensor::new(vec![1], vec![6.0], false);
        let actual = &a / &b;
        let expected = vec![5.0/6.0];
        assert_eq!(actual.data, expected);
        assert!(a.grad.is_none());
        assert!(b.grad.is_none());
        assert_eq!(actual.bwd(), Err(TensorError::NoGrad));
    }
    #[test]
    fn div_int() {
        // 0d
        let a = Tensor::new(vec![1], vec![50.0], false);
        let b = Tensor::new(vec![1], vec![65.0], false);
        let actual = &a / &b;
        let expected = vec![50.0/65.0];
        assert_eq!(actual.data, expected);
        assert!(a.grad.is_none());
        assert!(b.grad.is_none());
        assert_eq!(actual.bwd(), Err(TensorError::NoGrad));
    }
    #[test]
    #[should_panic(expected = "")]
    fn div_shape_mismatch() {
        // 0d
        let a = Tensor::new(vec![1], vec![5.0, 2.0], false);
        let b = Tensor::new(vec![3], vec![2.0, 3.0, -6.0], false);
        let _ = &a / &b;
    }
}

#[cfg(test)]
mod sum_test {
    use crate::error::TensorError;
    use crate::Tensor;
    use crate::tensor::Dtype;

    fn check_sum<'a, T: Dtype>(a: &'a Tensor<'a, T>, expected_data: T) {
        let actual = a.sum();
        assert_eq!(actual.item(), expected_data);

        actual.bwd().unwrap();
        let grad = a.grad.as_ref().unwrap().borrow();
        assert_eq!(*grad, vec![T::one(); a.shape.iter().product()]);
    }

    #[test]
    fn sum_with_grad() {
        // 0d
        let a = Tensor::new(vec![1], vec![5.0], true);
        let expected_data = a.data.iter().sum();
        check_sum(&a, expected_data);

        // 1d
        let a = Tensor::new(vec![3], vec![1.0, -1.0, 2.0], true);
        let expected_data = a.data.iter().sum();
        check_sum(&a, expected_data);

        // 2d
        let a = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0], true);
        let expected_data = a.data.iter().sum();
        check_sum(&a, expected_data);

        // 3d
        let a = Tensor::new(vec![2, 2, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], true);
        let expected_data = a.data.iter().sum();
        check_sum(&a, expected_data);

        // 4d
        let a = Tensor::new(vec![2, 2, 2, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], true);
        let expected_data = a.data.iter().sum();
        check_sum(&a, expected_data);
    }
    #[test]
    fn sum_no_grad() {
        // 0d
        let a = Tensor::new(vec![1], vec![5.0], false);
        let actual = a.sum();
        let expected = 5.0;
        assert_eq!(actual.item(), expected);
        assert!(a.grad.is_none());
        assert_eq!(actual.bwd(), Err(TensorError::NoGrad));
    }
}

impl<'a, T: Dtype> Tensor<'a, T> {
    fn _canary(&self, exp: T) -> T {
        let value = self.data[0];
        value.pow(exp) * value.exp()
    }
}
