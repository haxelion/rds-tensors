use std::iter::repeat;
use std::borrow::{Borrow,BorrowMut};
use std::ops::{Index, IndexMut};

pub mod types;
pub mod tindex;

use types::{RDSType, RDSTyped, CastTo, CastFrom, Complex, c32, c64};
use tindex::TIndex;

/***************************************************************************************************
                                               TRAITS
***************************************************************************************************/

pub trait Tensor<T: RDSTyped> : Sized {

    fn from_scalar<R: AsRef<[usize]>>(shape: R, s: T) -> Self;

    fn zeros<R: AsRef<[usize]>>(shape: R) -> Self {
        Self::from_scalar::<R>(shape, T::cast_from(0u8))
    }

    fn ones<R: AsRef<[usize]>>(shape: R) -> Self {
        Self::from_scalar::<R>(shape, T::cast_from(1u8))
    }

    fn from_tensor<W: Tensor<S>, S: RDSTyped + CastTo<T>>(tensor: &W) -> Self;

    fn from_slice<R: AsRef<[usize]>, S: AsRef<[T]>>(shape: R, slice: S) -> Self;

    fn from_boxed_slice<R: AsRef<[usize]>>(shape: R, data: Box<[T]>) -> Self;

    fn into_boxed_slice(self) -> Box<[T]>;

    fn dim(&self) -> usize;

    fn shape(&self) -> &[usize];

    fn strides(&self) -> &[usize]; 

    fn size(&self) -> usize;

    fn rds_type(&self) -> RDSType;

    fn get_raw_array(&self) -> &[T];

    fn get_raw_array_mut(&mut self) -> &mut [T];

    fn reshape_into_vector<R: AsRef<[usize]>>(self, shape: R) -> Vector<T> {
        return Vector::from_boxed_slice(shape, self.into_boxed_slice());
    }

    fn reshape_into_matrix<R: AsRef<[usize]>>(self, shape: R) -> Matrix<T> {
        return Matrix::from_boxed_slice(shape, self.into_boxed_slice());
    }

    fn reshape_into_tensor<R: AsRef<[usize]>>(self, shape: R) -> TensorN<T> {
        return TensorN::from_boxed_slice(shape, self.into_boxed_slice());
    }
}

fn shape_to_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides: Vec<usize> = repeat(0usize).take(shape.len()).collect();
    let mut size = 1;
    for i in 0..shape.len()  {
        strides[shape.len()-i-1] = size;
        size *= shape[shape.len()-i-1];
    }
    return strides;
}

/***************************************************************************************************
                                               VECTOR
***************************************************************************************************/

pub struct Vector<T: RDSTyped> {
    data: Box<[T]>,
    shape: Vec<usize>,
    strides: Vec<usize>
}

impl<T: RDSTyped> Tensor<T> for Vector<T> {
    fn from_scalar<R: AsRef<[usize]>>(shape: R, s: T) -> Self {
        let shape = shape.as_ref();
        assert!(shape.len() == 1, "Vector::from_scalar(): provided shape has more than one dimension");
        let data: Vec<T> = repeat(s).take(shape[0]).collect();
        Vector {
            data: data.into_boxed_slice(),
            shape: vec![shape[0]],
            strides: vec![1]
        }
    }

    fn from_slice<R: AsRef<[usize]>, S: AsRef<[T]>>(shape: R, slice: S) -> Self {
        return Self::from_boxed_slice(shape, slice.as_ref().to_vec().into_boxed_slice());
    }

    fn from_tensor<W: Tensor<S>, S: RDSTyped + CastTo<T>>(tensor: &W) -> Self {
        assert!(tensor.shape().iter().fold(tensor.dim(), |acc, &i| acc - (i == 1) as usize) <= 1, 
                "Vector::from_tensor(): provided tensor has more than one non-unit dimension");
        let data: Vec<T> = tensor.get_raw_array().iter().map(|&i| i.cast_to()).collect();
        Vector {
            shape: vec![data.len()],
            data: data.into_boxed_slice(),
            strides: vec![1]
        }
    }

    fn from_boxed_slice<R: AsRef<[usize]>>(shape: R, data: Box<[T]>) -> Self {
        let shape = shape.as_ref();
        assert!(shape.len() == 1, "Vector::from_boxed_slice(): provided shape has more than one dimension");
        assert!(shape[0] == data.len(), "Vector::from_boxed_slice(): provided shape and slice do not have the same number of elements");
        Vector {
            shape: vec![data.len()],
            data: data,
            strides: vec![1]
        }
    }

    fn into_boxed_slice(self) -> Box<[T]> {
        return self.data;
    }

    fn dim(&self) -> usize { 
        1 
    }

    fn shape(&self) -> &[usize] { 
        self.shape.as_slice()
    }

    fn strides(&self) -> &[usize] { 
        self.strides.as_slice()
    }

    fn size(&self) -> usize { 
        self.data.len() 
    }

    fn rds_type(&self) -> RDSType {
        T::rds_type()
    }

    fn get_raw_array(&self) -> &[T] {
        self.data.borrow()
    }

    fn get_raw_array_mut(&mut self) -> &mut [T] {
        self.data.borrow_mut()
    }
}

impl<I,T> Index<I> for Vector<T> where I: AsRef<[usize]>, T: RDSTyped {
    type Output = T;

    fn index(&self, i: I) -> &T {
        let i = i.as_ref();
        assert!(i.len() == 1, "Vector::index(): provided index has more than one dimension");
        return &self.data[i[0]];
    }
}

impl<I,T> IndexMut<I> for Vector<T> where I: AsRef<[usize]>, T: RDSTyped {
    fn index_mut(&mut self, i: I) -> &mut T {
        let i = i.as_ref();
        assert!(i.len() == 1, "Vector::index(): provided index_mut has more than one dimension");
        return &mut self.data[i[0]];
    }
}

/***************************************************************************************************
                                               MATRIX
***************************************************************************************************/

pub struct Matrix<T: RDSTyped> {
    data: Box<[T]>,
    shape: Vec<usize>,
    strides: Vec<usize>
}

impl<T: RDSTyped> Tensor<T> for Matrix<T> {
    fn from_scalar<R: AsRef<[usize]>>(shape: R, s: T) -> Self {
        let shape = shape.as_ref();
        assert!(shape.len() == 2, "Matrix::from_scalar(): provided shape is not two dimensional");
        let data: Vec<T> = repeat(s).take(shape[0]*shape[1]).collect();
        Matrix {
            data: data.into_boxed_slice(),
            shape: shape.to_vec(),
            strides: vec![shape[1], 1]
        }
    }

    fn from_tensor<W: Tensor<S>, S: RDSTyped + CastTo<T>>(tensor: &W) -> Self {
        let mut shape = Vec::<usize>::with_capacity(2);
        for l in tensor.shape() {
            if *l > 1 {
                shape.push(*l);
            }
        }
        assert!(shape.len() <= 2, "Matrix::from_tensor(): provided tensor has more than two non-unit dimensions");
        while shape.len() < 2 {
            shape.push(1);
        }
        let data: Vec<T> = tensor.get_raw_array().iter().map(|&i| i.cast_to()).collect();
        Matrix {
            strides: vec![shape[1], 1],
            shape: shape,
            data: data.into_boxed_slice()
        }
    }

    fn from_slice<R: AsRef<[usize]>, S: AsRef<[T]>>(shape: R, slice: S) -> Self {
        return Self::from_boxed_slice(shape, slice.as_ref().to_vec().into_boxed_slice());
    }

    fn from_boxed_slice<R: AsRef<[usize]>>(shape: R, data: Box<[T]>) -> Self {
        let shape = shape.as_ref();
        assert!(shape.len() == 2, "Matrix::from_boxed_slice(): provided shape has more than two dimensions");
        assert!(shape[0] * shape[1] == data.len(), "Matrix::from_boxed_slice(): provided data and shape does not have the same number of elements");
        Matrix {
            data: data,
            shape: shape.to_vec(),
            strides: vec![shape[1], 1]
        }
    }

    fn into_boxed_slice(self) -> Box<[T]> {
        return self.data;
    }

    fn dim(&self) -> usize { 
        2
    }

    fn shape(&self) -> &[usize] { 
        self.shape.as_slice()
    }

    fn strides(&self) -> &[usize] { 
        self.strides.as_slice()
    }

    fn size(&self) -> usize { 
        self.data.len() 
    }

    fn rds_type(&self) -> RDSType {
        T::rds_type()
    }

    fn get_raw_array(&self) -> &[T] {
        self.data.borrow()
    }

    fn get_raw_array_mut(&mut self) -> &mut [T] {
        self.data.borrow_mut()
    }
}

impl<I,T> Index<I> for Matrix<T> where I: AsRef<[usize]>, T: RDSTyped {
    type Output = T;

    fn index(&self, i: I) -> &T {
        let i = i.as_ref();
        assert!(i.len() == 2, "Matrix::index(): provided index is not two dimensional");
        let pos = i.to_pos(self.shape(), self.strides());
        return &self.data[pos];
    }
}

impl<I,T> IndexMut<I> for Matrix<T> where I: AsRef<[usize]>, T: RDSTyped {
    fn index_mut(&mut self, i: I) -> &mut T {
        let i = i.as_ref();
        assert!(i.len() == 2, "Matrix::index_mut(): provided index is not two dimensionsal");
        let pos = i.to_pos(self.shape(), self.strides());
        return &mut self.data[pos];
    }
}

/***************************************************************************************************
                                               TENSORN
***************************************************************************************************/

pub struct TensorN<T: RDSTyped> {
    data: Box<[T]>,
    shape: Vec<usize>,
    strides: Vec<usize>
}

impl<T: RDSTyped> Tensor<T> for TensorN<T> {
    fn from_scalar<R: AsRef<[usize]>>(shape: R, s: T) -> Self {
        let shape = shape.as_ref();
        let strides = shape_to_strides(shape);
        let size = strides[0] * shape[0];
        let data: Vec<T> = repeat(s).take(size).collect();
        TensorN {
            data: data.into_boxed_slice(),
            shape: shape.to_vec(),
            strides: strides
        }
    }

    fn from_tensor<W: Tensor<S>, S: RDSTyped + CastTo<T>>(tensor: &W) -> Self {
        let data: Vec<T> = tensor.get_raw_array().iter().map(|&i| i.cast_to()).collect();
        TensorN {
            shape: tensor.shape().to_vec(),
            data: data.into_boxed_slice(),
            strides: tensor.strides().to_vec()
        }
    }

    fn from_slice<R: AsRef<[usize]>, S: AsRef<[T]>>(shape: R, slice: S) -> Self {
        return Self::from_boxed_slice(shape, slice.as_ref().to_vec().into_boxed_slice());
    }

    fn from_boxed_slice<R: AsRef<[usize]>>(shape: R, data: Box<[T]>) -> Self {
        let shape = shape.as_ref();
        let strides = shape_to_strides(shape);
        assert!(strides[0] * shape[0] == data.len(), "TensorN::from_boxed_slice(): provided data and shape does not have the same number of elements");
        TensorN {
            data: data,
            shape: shape.to_vec(),
            strides: strides
        }
    }

    fn into_boxed_slice(self) -> Box<[T]> {
        return self.data;
    }

    fn dim(&self) -> usize { 
        self.shape.len()
    }

    fn shape(&self) -> &[usize] { 
        self.shape.as_slice()
    }

    fn strides(&self) -> &[usize] { 
        self.strides.as_slice()
    }

    fn size(&self) -> usize { 
        self.data.len() 
    }

    fn rds_type(&self) -> RDSType {
        T::rds_type()
    }

    fn get_raw_array(&self) -> &[T] {
        self.data.borrow()
    }

    fn get_raw_array_mut(&mut self) -> &mut [T] {
        self.data.borrow_mut()
    }
}

impl<I,T> Index<I> for TensorN<T> where I: AsRef<[usize]>, T: RDSTyped {
    type Output = T;

    fn index(&self, i: I) -> &T {
        let i = i.as_ref();
        assert!(i.len() == self.shape.len(), "TensorN::index(): provided index and this tensor have a different number of dimensions");
        let pos = i.to_pos(self.shape(), self.strides());
        return &self.data[pos];
    }
}

impl<I,T> IndexMut<I> for TensorN<T> where I: AsRef<[usize]>, T: RDSTyped {
    fn index_mut(&mut self, i: I) -> &mut T {
        let i = i.as_ref();
        assert!(i.len() == self.shape.len(), "TensorN::index(): provided index and this tensor have a different number of dimensions");
        let pos = i.to_pos(self.shape(), self.strides());
        return &mut self.data[pos];
    }
}


#[cfg(test)]
mod tests;
