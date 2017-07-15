use std::iter::repeat;
use std::borrow::{Borrow,BorrowMut};
use std::ops::{Index, IndexMut};

pub mod types;
pub mod tindex;

use types::{RDSType, RDSTyped, CastTo};
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

    fn effective_dim(&self) -> usize {
        self.shape().iter().fold(self.dim(), |acc, &i| acc - (i == 1) as usize)
    }

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

    fn insert<W: Tensor<T>>(&mut self, dim: usize, pos: usize, tensor: &W);
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
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>
}

impl<T: RDSTyped> Tensor<T> for Vector<T> {
    fn from_scalar<R: AsRef<[usize]>>(shape: R, s: T) -> Self {
        let shape = shape.as_ref();
        assert!(shape.len() == 1, "Vector::from_scalar(): provided shape has more than one dimension");
        let data: Vec<T> = repeat(s).take(shape[0]).collect();
        Vector {
            data: data,
            shape: vec![shape[0]],
            strides: vec![1]
        }
    }

    fn from_slice<R: AsRef<[usize]>, S: AsRef<[T]>>(shape: R, slice: S) -> Self {
        return Self::from_boxed_slice(shape, slice.as_ref().to_vec().into_boxed_slice());
    }

    fn from_tensor<W: Tensor<S>, S: RDSTyped + CastTo<T>>(tensor: &W) -> Self {
        assert!(tensor.effective_dim() <= 1,  "Vector::from_tensor(): provided tensor has more than one non-unit dimension");
        let data: Vec<T> = tensor.get_raw_array().iter().map(|&i| i.cast_to()).collect();
        Vector {
            shape: vec![data.len()],
            data: data,
            strides: vec![1]
        }
    }

    fn from_boxed_slice<R: AsRef<[usize]>>(shape: R, data: Box<[T]>) -> Self {
        let shape = shape.as_ref();
        assert!(shape.len() == 1, "Vector::from_boxed_slice(): provided shape has more than one dimension");
        assert!(shape[0] == data.len(), "Vector::from_boxed_slice(): provided shape and slice do not have the same number of elements");
        Vector {
            shape: vec![data.len()],
            data: data.into_vec(),
            strides: vec![1]
        }
    }

    fn into_boxed_slice(self) -> Box<[T]> {
        return self.data.into_boxed_slice();
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

    fn insert<W: Tensor<T>>(&mut self, dim: usize, pos: usize, tensor: &W) {
        assert!(dim == 0, "Vector::insert(): insertion dimension should be 0");
        assert!(pos <= self.shape[0], "Vector::insert(): insertion position is out of bound");
        assert!(tensor.dim() == 1 , "Vector::insert(): tensor to insert is not one dimensional");
        // Make self the right size
        self.data.reserve(tensor.size());
        // Write the extended part coming from tensor
        self.data.extend_from_slice(&tensor.get_raw_array()[(self.shape[0]-pos)..(tensor.size())]);
        // Write the extended part coming from self
        for i in pos..self.shape[0] {
            let t = self.data[i];
            self.data.push(t);
        }
        // Write the replaced part
        let tensor_data = tensor.get_raw_array();
        for i in pos..self.shape[0] {
            self.data[i] = tensor_data[i-pos];
        }
        self.shape[0] += tensor.size();
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
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>
}

impl<T: RDSTyped> Tensor<T> for Matrix<T> {
    fn from_scalar<R: AsRef<[usize]>>(shape: R, s: T) -> Self {
        let shape = shape.as_ref();
        assert!(shape.len() == 2, "Matrix::from_scalar(): provided shape is not two dimensional");
        let data: Vec<T> = repeat(s).take(shape[0]*shape[1]).collect();
        Matrix {
            data: data,
            shape: shape.to_vec(),
            strides: vec![shape[1], 1]
        }
    }

    fn from_tensor<W: Tensor<S>, S: RDSTyped + CastTo<T>>(tensor: &W) -> Self {
        assert!(tensor.effective_dim() <= 2, "Matrix::from_tensor(): provided tensor has more than two non-unit dimensions");
        let mut shape = Vec::<usize>::with_capacity(2);
        for l in tensor.shape() {
            if *l > 1 {
                shape.push(*l);
            }
        }
        while shape.len() < 2 {
            shape.push(1);
        }
        let data: Vec<T> = tensor.get_raw_array().iter().map(|&i| i.cast_to()).collect();
        Matrix {
            strides: vec![shape[1], 1],
            shape: shape,
            data: data
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
            data: data.into_vec(),
            shape: shape.to_vec(),
            strides: vec![shape[1], 1]
        }
    }

    fn into_boxed_slice(self) -> Box<[T]> {
        return self.data.into_boxed_slice();
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

    fn insert<W: Tensor<T>>(&mut self, dim: usize, pos: usize, tensor: &W) {
        assert!(dim <= 1, "Matrix::insert(): insertion dimension should be 0 or 1");
        assert!(pos <= self.shape[dim], "Matrix::insert(): insertion position is out of bound");
        assert!(tensor.dim() == 2 , "Matrix::insert(): tensor to insert is not two dimensional");
        // dim^1 will produce the dimension orthogonal to the insertion dimension (dim)
        assert!(self.shape[dim^1] == tensor.shape()[dim^1], "Matrix::insert(): tensor to insert has an incompatible shape");
        let mut idx_src_self = self.data.len();
        // Make self the right size
        self.data.reserve(tensor.size());
        for _ in 0..tensor.size() {
            self.data.push(T::cast_from(0u8));
        }
        // Back to front zipping parameter init
        let length = tensor.shape()[1];
        let count = tensor.shape()[0];
        let spacing = match dim {
            0 => 0,
            1 => self.shape[1],
            _ => unreachable!()
        };
        let offset = match dim {
            0 => (self.shape[0] - pos) * self.shape[1],
            1 => self.shape[1] - pos,
            _ => unreachable!()
        };
        // Back to front zipping
        let mut idx_dest = self.data.len();
        let mut idx_src_tensor = tensor.size();
        let tensor_data = tensor.get_raw_array();
        for _ in 0..offset {
            idx_dest -= 1;
            idx_src_self -= 1;
            self.data[idx_dest] = self.data[idx_src_self];
        }
        for _ in 0..length {
            idx_dest -= 1;
            idx_src_tensor -= 1;
            self.data[idx_dest] = tensor_data[idx_src_tensor];
        }
        for _ in 1..count {
            for _ in 0..spacing {
                idx_dest -= 1;
                idx_src_self -= 1;
                self.data[idx_dest] = self.data[idx_src_self];
            }
            for _ in 0..length {
                idx_dest -= 1;
                idx_src_tensor -= 1;
                self.data[idx_dest] = tensor_data[idx_src_tensor];
            }
        }
        self.shape[dim] += tensor.shape()[dim];
        self.strides = shape_to_strides(self.shape());
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
    data: Vec<T>,
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
            data: data,
            shape: shape.to_vec(),
            strides: strides
        }
    }

    fn from_tensor<W: Tensor<S>, S: RDSTyped + CastTo<T>>(tensor: &W) -> Self {
        let data: Vec<T> = tensor.get_raw_array().iter().map(|&i| i.cast_to()).collect();
        TensorN {
            shape: tensor.shape().to_vec(),
            data: data,
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
            data: data.into_vec(),
            shape: shape.to_vec(),
            strides: strides
        }
    }

    fn into_boxed_slice(self) -> Box<[T]> {
        return self.data.into_boxed_slice();
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

    fn insert<W: Tensor<T>>(&mut self, dim: usize, pos: usize, tensor: &W) {
        assert!(dim < self.dim(), "TensorN::insert(): insertion dimension should be 0 or 1");
        assert!(pos <= self.shape[dim], "TensorN::insert(): insertion position is out of bound");
        assert!(tensor.dim() == self.dim() , "TensorN::insert(): tensor has a different dimensionality than self");
        for i in 0..self.shape.len() {
            assert!((dim == i) || (self.shape[i] == tensor.shape()[i]), "TensorN::insert(): tensor shape doesn't match with self shape in the non-insertion dimensions");
        }

        let mut idx_src_self = self.data.len();
        // Make self the right size
        self.data.reserve(tensor.size());
        for _ in 0..tensor.size() {
            self.data.push(T::cast_from(0u8));
        }
        // Back to front zipping parameter init
        let length = tensor.shape()[dim..].iter().fold(1usize, |acc, &x| acc*x);
        let count = tensor.shape()[..dim].iter().fold(1usize, |acc, &x| acc*x);
        let spacing = self.shape[dim..].iter().fold(1usize, |acc, &x| acc*x);
        let offset = (self.shape[dim] - pos) * self.shape[dim+1..].iter().fold(1usize, |acc, &x| acc*x);
        // Back to front zipping
        let mut idx_dest = self.data.len();
        let mut idx_src_tensor = tensor.size();
        let tensor_data = tensor.get_raw_array();
        for _ in 0..offset {
            idx_dest -= 1;
            idx_src_self -= 1;
            self.data[idx_dest] = self.data[idx_src_self];
        }
        for _ in 0..length {
            idx_dest -= 1;
            idx_src_tensor -= 1;
            self.data[idx_dest] = tensor_data[idx_src_tensor];
        }
        for _ in 1..count {
            for _ in 0..spacing {
                idx_dest -= 1;
                idx_src_self -= 1;
                self.data[idx_dest] = self.data[idx_src_self];
            }
            for _ in 0..length {
                idx_dest -= 1;
                idx_src_tensor -= 1;
                self.data[idx_dest] = tensor_data[idx_src_tensor];
            }
        }
        self.shape[dim] += tensor.shape()[dim];
        self.strides = shape_to_strides(self.shape());
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
