use std::fmt::{Debug, Display};

use num_traits::Num;

pub trait Dtype: Num + Display + Debug + Copy + std::iter::Sum + std::ops::Neg<Output = Self> + std::ops::AddAssign +
    std::clone::Clone + std::cmp::PartialOrd {
    // TODO Add comments
    fn pow(&self, exp: Self) -> Self;
    fn exp(&self) -> Self;
    fn epsilon() -> Self;
    fn abs(&self) -> Self;

    fn almost_equal(lhs: &Self, rhs: &Self) -> bool {
        if lhs == rhs {
            return true;
        }
        else {
            (*lhs - *rhs).abs() < (*rhs * Self::epsilon()).abs()
        }
    }
}
impl Dtype for i8 {
    fn pow(&self, exp: Self) -> Self { i8::pow(*self, exp as u32) }
    fn exp(&self) -> Self { panic!("i8 does not support exp") } 
    fn epsilon() -> Self {
        0i8
    }
    fn abs(&self) -> Self { i8::abs(*self) }
}
impl Dtype for i16 {
    fn pow(&self, exp: Self) -> Self { i16::pow(*self, exp as u32) }
    fn exp(&self) -> Self { panic!("i16 does not support exp") } 
    fn epsilon() -> Self {
        0i16
    }
    fn abs(&self) -> Self { i16::abs(*self) }
}
impl Dtype for i32 {
    fn pow(&self, exp: Self) -> Self { i32::pow(*self, exp as u32) }
    fn exp(&self) -> Self { panic!("i32 does not support exp") } 
    fn epsilon() -> Self {
        0
    }
    fn abs(&self) -> Self { i32::abs(*self) }
}
// TODO Add f16 and bf16 support
impl Dtype for f32 {
    fn pow(&self, exp: Self) -> Self { f32::powf(*self, exp) } 
    fn exp(&self) -> Self { f32::exp(*self) }
    fn epsilon() -> Self {
        0.000001
    }
    fn abs(&self) -> Self { f32::abs(*self) }
}
impl Dtype for f64 {
    fn pow(&self, exp: Self) -> Self { f64::powf(*self, exp) }
    fn exp(&self) -> Self { f64::exp(*self) }
    fn epsilon() -> Self {
        0.00000000001
    }
    fn abs(&self) -> Self { f64::abs(*self) }
}