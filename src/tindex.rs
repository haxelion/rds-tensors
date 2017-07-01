/// A trait which adds indexing functions to simplify index manipulation.
pub trait TIndex {

    /// Increment an index in the row-major order.
    fn inc_ro(&mut self, shape : &[usize]);

    /// Decrement an index in the row-major order.
    fn dec_ro(&mut self, shape : &[usize]);

    /// Increment an index in the column-major order.
    fn inc_co(&mut self, shape : &[usize]);

    /// Decrement an index in the column-major order.
    fn dec_co(&mut self, shape : &[usize]);

    /// Return true if the index is zero for all dimensions.
    fn is_zero(&mut self) -> bool;
    
    /// Compute the resulting position in a linear storage array.
    fn to_pos(&self, shape : &[usize], strides : &[usize]) -> usize;
}

impl TIndex for [usize] {

    fn inc_ro(&mut self, shape : &[usize]) {
        let mut i = self.len();
        while i > 0 {
            self[i-1] += 1;
            if self[i-1] >= shape[i-1] {
                self[i-1] = 0;
                i -= 1;
            }
            else {
                break;
            }
        }
    }

    fn dec_ro(&mut self, shape : &[usize]) {
        let mut i = self.len();
        while i > 0 {
            if self[i-1] == 0 {
                self[i-1] = shape[i-1] - 1;
                i -= 1;
            }
            else {
                self[i-1] -= 1;
                break;
            }
        }
    }

    fn inc_co(&mut self, shape : &[usize]) {
        let mut i = 0;
        while i < self.len() {
            self[i] += 1;
            if self[i] >= shape[i] {
                self[i] = 0;
                i += 1;
            }
            else {
                break;
            }
        }
    }

    fn dec_co(&mut self, shape : &[usize]) {
        let mut i = 0;
        while i < self.len() {
            if self[i] ==  0 {
                self[i] = shape[i] - 1;
                i += 1;
            }
            else {
                self[i] -= 1;
                break;
            }
        }
    }

    fn is_zero(&mut self) -> bool {
        for i in 0..self.len() {
            if self[i] != 0 {
                return false;
            }
        }
        return true;
    }

    fn to_pos(&self, shape : &[usize], strides : &[usize]) -> usize {
        let mut pos = 0usize;
        for i in 0..self.len() {
            assert!(self[i] < shape[i], "TIndex::to_pos(): idx is out of bound.");
            pos += self[i] * strides[i];
        }
        return pos;
    }
}
