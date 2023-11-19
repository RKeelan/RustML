use std::fmt;

#[derive(Debug)]
pub enum TensorError {
    NoGrad,
    IncompatibleShape
}
impl std::error::Error for TensorError {}
impl fmt::Display for TensorError {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    match self {
        TensorError::NoGrad => write!(f, "Tensor had no Gradient"),
        TensorError::IncompatibleShape => write!(f, "Tensor shape(s) were imcompatible"),
    }
  }
}
impl PartialEq for TensorError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TensorError::NoGrad, TensorError::NoGrad) => true,
            (TensorError::IncompatibleShape, TensorError::IncompatibleShape) => true,
            _ => false
        }
    }
}