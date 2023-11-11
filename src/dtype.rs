use std::fmt::{Debug, Display};

use num_traits::Num;

pub trait Dtype: Num + Display + Debug + Copy + std::ops::AddAssign + std::clone::Clone {
    fn pow(&self, exp: Self) -> Self;
    fn exp(&self) -> Self;
}
impl Dtype for i8 {
    fn pow(&self, exp: Self) -> Self { i8::pow(*self, exp as u32) }
    fn exp(&self) -> Self { i8::exp(self) }
}
impl Dtype for i16 {
    fn pow(&self, exp: Self) -> Self { i16::pow(*self, exp as u32) }
    fn exp(&self) -> Self { i16::exp(self) }
}
impl Dtype for i32 {
    fn pow(&self, exp: Self) -> Self { i32::pow(*self, exp as u32) }
    fn exp(&self) -> Self { i32::exp(self) }
}
// TODO Add f16 and bf16 support
impl Dtype for f32 {
    fn pow(&self, exp: Self) -> Self { f32::powf(*self, exp) } 
    fn exp(&self) -> Self { f32::exp(*self) }
}
impl Dtype for f64 {
    fn pow(&self, exp: Self) -> Self { f64::powf(*self, exp) }
    fn exp(&self) -> Self { f64::exp(*self) }
}