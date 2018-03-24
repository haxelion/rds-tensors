use std::iter::repeat;
use std::borrow::{Borrow,BorrowMut};
use std::fmt;
use std::ops::{Index, IndexMut, Range};

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

    fn extract<R: AsRef<[Range<usize>]>>(&self, bounds: R) -> Self;

    fn remove<R: AsRef<[Range<usize>]>>(&mut self, bounds: R);

    fn assign<R: AsRef<[Range<usize>]>, W: Tensor<T>>(&mut self, bounds: R, tensor: &W);

    fn right_split(&mut self, dim: usize, pos: usize) -> Self {
        debug_assert!(dim < self.dim(), "TensorN::right_split(): splitting dimension is invalid");
        debug_assert!(pos <= self.shape()[dim], "TensorN::right_split(): splitting position is out of bound");
        let mut bounds: Vec<Range<usize>> = self.shape().iter().map(|&x| 0..x).collect();
        bounds[dim].start = pos;
        let right = self.extract(&bounds);
        self.remove(&bounds);
        return right;
    }

    fn left_split(&mut self, dim: usize, pos: usize) -> Self {
        debug_assert!(dim < self.dim(), "TensorN::left_split(): splitting dimension is invalid");
        debug_assert!(pos <= self.shape()[dim], "TensorN::left_split(): splitting position is out of bound");
        let mut bounds: Vec<Range<usize>> = self.shape().iter().map(|&x| 0..x).collect();
        bounds[dim].end = pos;
        let left = self.extract(&bounds);
        self.remove(&bounds);
        return left;
    }


    fn transpose(&mut self);
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
        debug_assert!(shape.len() == 1, "Vector::from_scalar(): provided shape has more than one dimension");
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
        debug_assert!(tensor.effective_dim() <= 1,  "Vector::from_tensor(): provided tensor has more than one non-unit dimension");
        let data: Vec<T> = tensor.get_raw_array().iter().map(|&i| i.cast_to()).collect();
        Vector {
            shape: vec![data.len()],
            data: data,
            strides: vec![1]
        }
    }

    fn from_boxed_slice<R: AsRef<[usize]>>(shape: R, data: Box<[T]>) -> Self {
        let shape = shape.as_ref();
        debug_assert!(shape.len() == 1, "Vector::from_boxed_slice(): provided shape has more than one dimension");
        debug_assert!(shape[0] == data.len(), "Vector::from_boxed_slice(): provided shape and slice do not have the same number of elements");
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
        debug_assert!(dim == 0, "Vector::insert(): insertion dimension should be 0");
        debug_assert!(pos <= self.shape[0], "Vector::insert(): insertion position is out of bound");
        debug_assert!(tensor.dim() == 1 , "Vector::insert(): tensor to insert is not uni-dimensional");
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

    fn extract<R: AsRef<[Range<usize>]>>(&self, bounds: R) -> Self {
        let bounds = bounds.as_ref();
        debug_assert!(bounds.len() == 1, "Vector::extract(): bounds should be uni-dimensional");
        debug_assert!(bounds[0].start <= self.shape[0] && bounds[0].end <= self.shape[0], "Vector::extract(): bounds are out of range");
        debug_assert!(bounds[0].start <= bounds[0].end, "Vector::extract(): range start is after range end");
        return Vector::<T>::from_slice([bounds[0].end - bounds[0].start], &self.data[bounds[0].clone()]);
    }

    fn remove<R: AsRef<[Range<usize>]>>(&mut self, bounds: R) {
        let bounds = bounds.as_ref();
        debug_assert!(bounds.len() == 1, "Vector::remove(): bounds should be uni-dimensional");
        debug_assert!(bounds[0].start <= self.shape[0] && bounds[0].end <= self.shape[0], "Vector::remove(): bounds are out of range");
        debug_assert!(bounds[0].start <= bounds[0].end, "Vector::remove(): range start is after range end");
        let delta = bounds[0].end - bounds[0].start;
        for i in bounds[0].end..self.shape[0] {
            self.data[i - delta] = self.data[i]
        }
        self.shape[0] -= delta;
        self.data.truncate(self.shape[0]);
    }

    fn assign<R: AsRef<[Range<usize>]>, W: Tensor<T>>(&mut self, bounds: R, tensor: &W) {
        let bounds = bounds.as_ref();
        debug_assert!(bounds.len() == 1, "Vector::assign(): bounds should be uni-dimensional");
        debug_assert!(tensor.dim() == 1, "Vector::assign(): tensor should be uni-dimensional");
        debug_assert!(bounds[0].end - bounds[0].start <= tensor.shape()[0], "Vector::assign(): bounds are out of range");
        debug_assert!(bounds[0].start <= self.shape[0] && bounds[0].end <= self.shape[0], "Vector::assign(): bounds are out of range");
        debug_assert!(bounds[0].start <= bounds[0].end, "Vector::assign(): range start is after range end");
        for i in 0..(bounds[0].end - bounds[0].start) {
            self.data[bounds[0].start + i] = tensor.get_raw_array()[i];
        }
    }

    fn transpose(&mut self) {
        debug_assert!(false, "Vector::transpose(): transposition of a vector has no effect");
    }
}

impl<I,T> Index<I> for Vector<T> where I: AsRef<[usize]>, T: RDSTyped {
    type Output = T;

    fn index(&self, i: I) -> &T {
        let i = i.as_ref();
        debug_assert!(i.len() == 1, "Vector::index(): provided index has more than one dimension");
        return &self.data[i[0]];
    }
}

impl<I,T> IndexMut<I> for Vector<T> where I: AsRef<[usize]>, T: RDSTyped {
    fn index_mut(&mut self, i: I) -> &mut T {
        let i = i.as_ref();
        debug_assert!(i.len() == 1, "Vector::index(): provided index_mut has more than one dimension");
        return &mut self.data[i[0]];
    }
}

impl<T: RDSTyped> fmt::Display for Vector<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..self.shape[0] {
            write!(f, "{}", self.data[i])?;
            if i != self.shape[0]-1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
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
        debug_assert!(shape.len() == 2, "Matrix::from_scalar(): provided shape is not two dimensional");
        let data: Vec<T> = repeat(s).take(shape[0]*shape[1]).collect();
        Matrix {
            data: data,
            shape: shape.to_vec(),
            strides: vec![shape[1], 1]
        }
    }

    fn from_tensor<W: Tensor<S>, S: RDSTyped + CastTo<T>>(tensor: &W) -> Self {
        debug_assert!(tensor.effective_dim() <= 2, "Matrix::from_tensor(): provided tensor has more than two non-unit dimensions");
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
        debug_assert!(shape.len() == 2, "Matrix::from_boxed_slice(): provided shape is not two dimensional");
        debug_assert!(shape[0] * shape[1] == data.len(), "Matrix::from_boxed_slice(): provided data and shape does not have the same number of elements");
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
        debug_assert!(dim <= 1, "Matrix::insert(): insertion dimension should be 0 or 1");
        debug_assert!(pos <= self.shape[dim], "Matrix::insert(): insertion position is out of bound");
        debug_assert!(tensor.dim() == 2 , "Matrix::insert(): tensor to insert is not two dimensional");
        // dim^1 will produce the dimension orthogonal to the insertion dimension (dim)
        debug_assert!(self.shape[dim^1] == tensor.shape()[dim^1], "Matrix::insert(): tensor to insert has an incompatible shape");
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
        let mut idx_dst = self.data.len();
        let mut idx_src_tensor = tensor.size();
        let tensor_data = tensor.get_raw_array();
        for _ in 0..offset {
            idx_dst -= 1;
            idx_src_self -= 1;
            self.data[idx_dst] = self.data[idx_src_self];
        }
        for _ in 0..length {
            idx_dst -= 1;
            idx_src_tensor -= 1;
            self.data[idx_dst] = tensor_data[idx_src_tensor];
        }
        for _ in 1..count {
            for _ in 0..spacing {
                idx_dst -= 1;
                idx_src_self -= 1;
                self.data[idx_dst] = self.data[idx_src_self];
            }
            for _ in 0..length {
                idx_dst -= 1;
                idx_src_tensor -= 1;
                self.data[idx_dst] = tensor_data[idx_src_tensor];
            }
        }
        self.shape[dim] += tensor.shape()[dim];
        self.strides = shape_to_strides(self.shape());
    }

    fn extract<R: AsRef<[Range<usize>]>>(&self, bounds: R) -> Self {
        let bounds = bounds.as_ref();
        debug_assert!(bounds.len() == 2, "Matrix::extract(): bounds should be two dimensional");
        debug_assert!(bounds[0].start <= self.shape[0] && bounds[0].end <= self.shape[0], "Matrix::extract(): bounds are out of range");
        debug_assert!(bounds[1].start <= self.shape[1] && bounds[1].end <= self.shape[1], "Matrix::extract(): bounds are out of range");
        debug_assert!(bounds[0].start <= bounds[0].end, "Matrix::extract(): range start is after range end");
        debug_assert!(bounds[1].start <= bounds[1].end, "Matrix::extract(): range start is after range end");
        // New exact allocation
        let mut new_data = Vec::<T>::with_capacity((bounds[0].end - bounds[0].start) * (bounds[1].end - bounds[1].start));
        // Striding parameters
        let mut idx_src = bounds[0].start * self.strides[0] + bounds[1].start;
        let length = bounds[1].end - bounds[1].start;
        // Extract row by row
        for _ in bounds[0].clone() {
            new_data.extend_from_slice(&self.data[idx_src..idx_src+length]);
            idx_src += self.strides[0];
        }
        return Matrix::<T>::from_boxed_slice([bounds[0].end - bounds[0].start, bounds[1].end - bounds[1].start], new_data.into_boxed_slice());
    }

    fn remove<R: AsRef<[Range<usize>]>>(&mut self, bounds: R) {
        let bounds = bounds.as_ref();
        let mut removed_dim: Option<usize> = None;
        debug_assert!(bounds.len() == 2, "Matrix::remove(): bounds should be two dimensional");
        for i in 0..2 {
            debug_assert!(bounds[i].start <= self.shape[i] && bounds[i].end <= self.shape[i], "Matrix::remove(): bounds are out of range");
            debug_assert!(bounds[i].start <= bounds[i].end, "Matrix::remove(): range start is after range end");
            if (bounds[i].end - bounds[i].start) != self.shape[i] {
                debug_assert!(removed_dim.is_none(), "Matrix::remove(): bounds should match all dimensions but one");
                removed_dim = Some(i);
            }
        }
        debug_assert!(removed_dim.is_some(), "Matrix::remove(): cannot remove the entire matrix");
        let removed_dim = removed_dim.unwrap();
        // Front to back unzipping parameter init
        let length = bounds[1].end - bounds[1].start;
        let count = bounds[0].end - bounds[0].start;
        let spacing = self.strides[0] - length;
        let offset = bounds[0].start * self.strides[0] + bounds[1].start * self.strides[1];
        // Front to back in-place unzipping
        let mut idx_dst = offset;
        let mut idx_src = offset + length;
        for _ in 1..count {
            for _ in 0..spacing {
                self.data[idx_dst] = self.data[idx_src];
                idx_dst += 1;
                idx_src += 1;
            }
            for _ in 0..length {
                idx_src += 1;
            }
        }
        while idx_src < self.data.len() {
            self.data[idx_dst] = self.data[idx_src];
            idx_dst += 1;
            idx_src += 1;
        }
        self.shape[removed_dim] -= bounds[removed_dim].end - bounds[removed_dim].start;
        self.strides = shape_to_strides(self.shape());
        self.data.truncate(self.shape[0] * self.shape[1]);
    }

    fn assign<R: AsRef<[Range<usize>]>, W: Tensor<T>>(&mut self, bounds: R, tensor: &W) {
        let bounds = bounds.as_ref();
        debug_assert!(bounds.len() == 2, "Matrix::assign(): bounds should be two dimensional");
        debug_assert!(tensor.dim() == 2, "Matrix::assign(): tensor should be two dimensional");
        debug_assert!(bounds[0].start <= bounds[0].end, "Matrix::assign(): range start is after range end");
        debug_assert!(bounds[1].start <= bounds[1].end, "Matrix::assign(): range start is after range end");
        debug_assert!(bounds[0].end - bounds[0].start <= tensor.shape()[0], "Matrix::assign(): bounds are out of range");
        debug_assert!(bounds[1].end - bounds[1].start <= tensor.shape()[1], "Matrix::assign(): bounds are out of range");
        debug_assert!(bounds[0].start <= self.shape[0] && bounds[0].end <= self.shape[0], "Matrix::assign(): bounds are out of range");
        debug_assert!(bounds[1].start <= self.shape[1] && bounds[1].end <= self.shape[1], "Matrix::assign(): bounds are out of range");
        for i in 0..(bounds[0].end - bounds[0].start) {
            let self_idx = (bounds[0].start + i) * self.strides[0];
            let tensor_idx = i * tensor.strides()[0];
            for j in 0..(bounds[1].end - bounds[1].start) {
                self.data[self_idx + bounds[1].start + j] = tensor.get_raw_array()[tensor_idx + j];
            }
        }
    }
    
    fn transpose(&mut self) {
        let mut init_idx = 0;
        let mut new_shape = self.shape.clone();
        new_shape.reverse();
        let new_strides = shape_to_strides(&new_shape);
        'outer: while init_idx < self.data.len() {
            let start_idx = init_idx;
            let mut idx = start_idx;
            // Check walk
            loop {
                // Compute next transposition transformation
                idx = (idx % self.strides[0]) * new_strides[0] + (idx / self.strides[0]);
                // Cycled ended
                if idx == start_idx {
                    break;
                }
                // Optimization to reduce the large loop iteration numbers
                else if idx == init_idx + 1 {
                    init_idx += 1;
                }
                // This cycle goes through an idx < start_idx -> we have already gone through it
                else if idx < start_idx {
                    init_idx += 1;
                    continue 'outer;
                }
            }
            // Execute walk
            idx = start_idx;
            let mut cycle_carry = self.data[idx];
            loop {
                // Compute next transposition transformation
                idx = (idx % self.strides[0]) * new_strides[0] + (idx / self.strides[0]);
                // Swap cycle_carry <-> self.data[idx]
                let t = self.data[idx];
                self.data[idx] = cycle_carry;
                cycle_carry = t;
                // Cycled ended
                if idx == start_idx {
                    break;
                }
                // Optimization to reduce the large loop iteration numbers
                else if idx == init_idx + 1 {
                    init_idx += 1;
                }
            }
            init_idx += 1;
        }
        self.shape = new_shape;
        self.strides = new_strides;
    }
}

impl<I,T> Index<I> for Matrix<T> where I: AsRef<[usize]>, T: RDSTyped {
    type Output = T;

    fn index(&self, i: I) -> &T {
        let i = i.as_ref();
        debug_assert!(i.len() == 2, "Matrix::index(): provided index is not two dimensional");
        let pos = i.to_pos(self.shape(), self.strides());
        return &self.data[pos];
    }
}

impl<I,T> IndexMut<I> for Matrix<T> where I: AsRef<[usize]>, T: RDSTyped {
    fn index_mut(&mut self, i: I) -> &mut T {
        let i = i.as_ref();
        debug_assert!(i.len() == 2, "Matrix::index_mut(): provided index is not two dimensionsal");
        let pos = i.to_pos(self.shape(), self.strides());
        return &mut self.data[pos];
    }
}

impl<T: RDSTyped> fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..self.shape[0] {
            write!(f, "\t[")?;
            for j in 0..self.shape[1] {
                write!(f, "{}", self[[i,j]])?;
                if j != self.shape[1]-1 {
                    write!(f, ", ")?;
                }
            }
            if i != self.shape[0]-1 {
                writeln!(f, "],")?;
            }
            else {
                writeln!(f, "]")?;
            }
        }
        write!(f, "]")
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
        debug_assert!(shape.len() > 0, "TensorN::from_scalar(): shape should have a least one dimension");
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
        debug_assert!(shape.len() > 0, "TensorN::from_boxed_slice(): shape should have a least one dimension");
        let strides = shape_to_strides(shape);
        debug_assert!(strides[0] * shape[0] == data.len(), "TensorN::from_boxed_slice(): provided data and shape does not have the same number of elements");
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
        debug_assert!(dim < self.dim(), "TensorN::insert(): insertion dimension should be 0 or 1");
        debug_assert!(pos <= self.shape[dim], "TensorN::insert(): insertion position is out of bound");
        debug_assert!(tensor.dim() == self.dim() , "TensorN::insert(): tensor has a different dimensionality than self");
        for i in 0..self.shape.len() {
            debug_assert!((dim == i) || (self.shape[i] == tensor.shape()[i]), "TensorN::insert(): tensor shape doesn't match with self shape in the non-insertion dimensions");
        }

        let mut idx_src_self = self.data.len();
        // Make self the right size
        self.data.reserve(tensor.size());
        for _ in 0..tensor.size() {
            self.data.push(T::cast_from(0u8));
        }
        // Back to front zipping parameter init
        let length = tensor.shape()[dim..].iter().fold(1usize, |acc, x| acc*x);
        let count = tensor.shape()[..dim].iter().fold(1usize, |acc, x| acc*x);
        let spacing = self.shape[dim..].iter().fold(1usize, |acc, x| acc*x);
        let offset = (self.shape[dim] - pos) * self.shape[dim+1..].iter().fold(1usize, |acc, x| acc*x);
        // Back to front zipping
        let mut idx_dst = self.data.len();
        let mut idx_src_tensor = tensor.size();
        let tensor_data = tensor.get_raw_array();
        for _ in 0..offset {
            idx_dst -= 1;
            idx_src_self -= 1;
            self.data[idx_dst] = self.data[idx_src_self];
        }
        for _ in 0..length {
            idx_dst -= 1;
            idx_src_tensor -= 1;
            self.data[idx_dst] = tensor_data[idx_src_tensor];
        }
        for _ in 1..count {
            for _ in 0..spacing {
                idx_dst -= 1;
                idx_src_self -= 1;
                self.data[idx_dst] = self.data[idx_src_self];
            }
            for _ in 0..length {
                idx_dst -= 1;
                idx_src_tensor -= 1;
                self.data[idx_dst] = tensor_data[idx_src_tensor];
            }
        }
        self.shape[dim] += tensor.shape()[dim];
        self.strides = shape_to_strides(self.shape());
    }

    fn extract<R: AsRef<[Range<usize>]>>(&self, bounds: R) -> Self {
        let bounds = bounds.as_ref();
        debug_assert!(bounds.len() == self.dim(), "TensorN::extract(): bounds and self have different dimensionality");
        for i in 0..self.dim() {
            debug_assert!(bounds[i].start <= self.shape[i] && bounds[i].end <= self.shape[i], "TensorN::extract(): bounds are out of range");
            debug_assert!(bounds[i].start <= bounds[i].end, "TensorN::extract(): range start is after range end");
        }
        let size = bounds.iter().fold(1usize, |acc, b| acc * (b.end - b.start));
        let mut new_data = Vec::<T>::with_capacity(size);
        if size > 0 {
            //let mut new_strides = shape_to_strides(&new_shape);
            let mut idx: Vec<usize> = bounds.iter().map(|b| b.start).collect();
            let length = bounds[self.dim()-1].end - bounds[self.dim()-1].start;
            // Extract row by row
            loop {
                let start_pos = idx.to_pos(&self.shape[..], &self.strides[..]);
                new_data.extend_from_slice(&self.data[start_pos..start_pos+length]);
                // Row order increment but within bounds
                // Start from the before last dimension (because we do row by row copy)
                let mut i = self.dim()-1; 
                while i > 0 {
                    idx[i-1] += 1;
                    if idx[i-1] >= bounds[i-1].end {
                        idx[i-1] = bounds[i-1].start;
                        i -= 1;
                    }
                    else {
                        break;
                    }
                }
                // Complete overflow of the idx, we inserted everything
                if i == 0 {
                    break;
                }
            }
        }
        let new_shape : Vec<usize> = bounds.iter().map(|b| b.end - b.start).collect();
        return TensorN::<T>::from_boxed_slice(new_shape, new_data.into_boxed_slice());
    }

    fn remove<R: AsRef<[Range<usize>]>>(&mut self, bounds: R) {
        let bounds = bounds.as_ref();
        let mut removed_dim: Option<usize> = None;
        debug_assert!(bounds.len() == self.dim(), "TensorN::remove(): bounds and self should have the same dimensionality");
        for i in 0..self.dim() {
            debug_assert!(bounds[i].start <= self.shape[i] && bounds[i].end <= self.shape[i], "TensorN::remove(): bounds are out of range");
            debug_assert!(bounds[i].start <= bounds[i].end, "TensorN::remove(): range start is after range end");
            if (bounds[i].end - bounds[i].start) != self.shape[i] {
                debug_assert!(removed_dim.is_none(), "TensorN::remove(): bounds should match all dimensions but one");
                removed_dim = Some(i);
            }
        }
        debug_assert!(removed_dim.is_some(), "TensorN::remove(): cannot remove the entire tensor");
        let removed_dim = removed_dim.unwrap();
        // Front to back unzipping parameter init
        let length = bounds[removed_dim..].iter().fold(1usize, |acc, x| acc*(x.end - x.start));
        let count = bounds[..removed_dim].iter().fold(1usize, |acc, x| acc*(x.end - x.start));
        let spacing = self.shape[removed_dim..].iter().fold(1usize, |acc, x| acc * x) - length;
        let offset = bounds.iter().zip(self.strides.iter()).fold(0usize, |acc, (b, s)| acc + s * b.start);
        // Front to back unzipping
        let mut idx_dst = offset;
        let mut idx_src = offset + length;
        for _ in 1..count {
            for _ in 0..spacing {
                self.data[idx_dst] = self.data[idx_src];
                idx_dst += 1;
                idx_src += 1;
            }
            for _ in 0..length {
                idx_src += 1;
            }
        }
        while idx_src < self.data.len() {
            self.data[idx_dst] = self.data[idx_src];
            idx_dst += 1;
            idx_src += 1;
        }
        self.shape[removed_dim] -= bounds[removed_dim].end - bounds[removed_dim].start;
        self.strides = shape_to_strides(self.shape());
        self.data.truncate(self.shape.iter().fold(1usize, |acc, x| acc * x));
    }

    fn assign<R: AsRef<[Range<usize>]>, W: Tensor<T>>(&mut self, bounds: R, tensor: &W) {
        let bounds = bounds.as_ref();
        debug_assert!(bounds.len() == self.dim(), "TensorN::assign(): bounds should be two dimensional");
        debug_assert!(tensor.dim() == self.dim(), "TensorN::assign(): tensor should be two dimensional");
        for i in 0..self.dim() {
            debug_assert!(bounds[i].start <= bounds[i].end, "TensorN::assign(): range start is after range end");
            debug_assert!(bounds[i].end - bounds[i].start <= tensor.shape()[i], "TensorN::assign(): bounds are out of range");
            debug_assert!(bounds[i].start <= self.shape[i] && bounds[i].end <= self.shape[i], "TensorN::assign(): bounds are out of range");
        }
        let bounds_shape: Vec<usize> = bounds.iter().map(|x| x.end - x.start).collect();
        let mut tensor_idx: Vec<usize> = repeat(0).take(self.dim()).collect();
        let mut self_idx = tensor_idx.clone();

        loop {
            for i in 0..self.dim() {
                self_idx[i] = tensor_idx[i] + bounds[i].start;
            }
            self[&self_idx] = tensor.get_raw_array()[tensor_idx.to_pos(tensor.shape(), tensor.strides())];
            tensor_idx.inc_ro(&bounds_shape);
            if tensor_idx.is_zero() {
                break;
            }
        }
    }

    fn transpose(&mut self) {
        let mut init_idx = 0;
        let mut new_shape = self.shape.clone();
        new_shape.reverse();
        let new_strides = shape_to_strides(&new_shape);
        'outer: while init_idx < self.data.len() {
            let start_idx = init_idx;
            let mut idx = start_idx;
            // Check walk
            loop {
                // Compute next transposition transformation
                let mut next_idx = 0;
                for i in 0..self.strides.len() {
                    next_idx += (idx / self.strides[i]) * new_strides[new_strides.len() - i - 1];
                    idx = idx % self.strides[i];
                }
                idx = next_idx;
                // Cycled ended
                if idx == start_idx {
                    break;
                }
                // Optimization to reduce the large loop iteration numbers
                else if idx == init_idx + 1 {
                    init_idx += 1;
                }
                // This cycle goes through an idx < start_idx -> we have already gone through it
                else if idx < start_idx {
                    init_idx += 1;
                    continue 'outer;
                }
            }
            // Execute walk
            idx = start_idx;
            let mut cycle_carry = self.data[idx];
            loop {
                // Compute next transposition transformation
                let mut next_idx = 0;
                for i in 0..self.strides.len() {
                    next_idx += (idx / self.strides[i]) * new_strides[new_strides.len() - i - 1];
                    idx = idx % self.strides[i];
                }
                idx = next_idx;
                // Swap cycle_carry <-> self.data[idx]
                let t = self.data[idx];
                self.data[idx] = cycle_carry;
                cycle_carry = t;
                // Cycled ended
                if idx == start_idx {
                    break;
                }
                // Optimization to reduce the large loop iteration numbers
                else if idx == init_idx + 1 {
                    init_idx += 1;
                }
            }
            init_idx += 1;
        }
        self.shape = new_shape;
        self.strides = new_strides;
    }
}

impl<I,T> Index<I> for TensorN<T> where I: AsRef<[usize]>, T: RDSTyped {
    type Output = T;

    fn index(&self, i: I) -> &T {
        let i = i.as_ref();
        debug_assert!(i.len() == self.shape.len(), "TensorN::index(): provided index and this tensor have a different number of dimensions");
        let pos = i.to_pos(self.shape(), self.strides());
        return &self.data[pos];
    }
}

impl<I,T> IndexMut<I> for TensorN<T> where I: AsRef<[usize]>, T: RDSTyped {
    fn index_mut(&mut self, i: I) -> &mut T {
        let i = i.as_ref();
        debug_assert!(i.len() == self.shape.len(), "TensorN::index(): provided index and this tensor have a different number of dimensions");
        let pos = i.to_pos(self.shape(), self.strides());
        return &mut self.data[pos];
    }
}

impl<T: RDSTyped> fmt::Display for TensorN<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut index: Vec<usize> = repeat(0).take(self.dim()).collect();
        // Down the braces
        for i in 0..self.dim() {
            for _j in 0..i {
                write!(f, "\t")?;
            }
            if i != self.dim() - 1 {
                writeln!(f, "[")?;
            }
            else {
                write!(f, "[")?;
            }
        }
        loop {
            // Print one element
            write!(f, "{}", self.data[index.to_pos(&self.shape, &self.strides)])?;
            // Index increment
            let mut i = self.dim();
            while i > 0 {
                index[i-1] += 1;
                if index[i-1] >= self.shape[i-1] {
                    index[i-1] = 0;
                    i -= 1;
                }
                else {
                    break;
                }
            }
            // End of the tensor, break out of the loop
            if i == 0 {
                break;
            }
            // If we overflow, print some braces
            if i < self.dim() {
                // Up the braces
                for j in 0..(self.dim() - i) {
                    // No tab for the first row
                    if j > 0 {
                        for _k in 0..(self.dim() - j - 1) {
                            write!(f, "\t")?;
                        }
                    }
                    if j == self.dim() - i - 1 {
                        writeln!(f, "],")?;
                    }
                    else {
                        writeln!(f, "]")?;
                    }
                }
                // Down the braces
                for j in i..self.dim() {
                    for _k in 0..j {
                        write!(f, "\t")?;
                    }
                    if j != self.dim() - 1 {
                        writeln!(f, "[")?;
                    }
                    else {
                        write!(f, "[")?;
                    }
                }
            }
            else {
                write!(f, ", ")?;
            }
        }
        for i in 0..self.dim() {
            if i > 0 {
                for _j in 0..(self.dim() - i - 1) {
                    write!(f, "\t")?;
                }
            }
            writeln!(f, "]")?;
        }
        return Ok(());
    }
}


#[cfg(test)]
mod tests;
