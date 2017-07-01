use Tensor;
use Vector;
use Matrix;
use TensorN;
use types::c32;
use types::c64;

#[test]
fn zeros() {
    let n = 10;
    let v = Vector::<f32>::zeros([n]);
    for i in 0..n {
        assert!(v[[i]] == 0.0);
    }
    let m = Matrix::<f64>::zeros([n,n]);
    for i in 0..n {
        for j in 0..n {
            assert!(m[[i,j]] == 0.0);
        }
    }
    let t = TensorN::<c32>::zeros([n,n,n]);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                assert!(t[[i,j,k]] == c32::new(0.0, 0.0));
            }
        }
    }
}

#[test]
fn ones() {
    let n = 10;
    let v = Vector::<u32>::ones([n]);
    for i in 0..n {
        assert!(v[[i]] == 1);
    }
    let m = Matrix::<u64>::ones([n,n]);
    for i in 0..n {
        for j in 0..n {
            assert!(m[[i,j]] == 1);
        }
    }
    let t = TensorN::<c64>::ones([n,n,n]);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                assert!(t[[i,j,k]] == c64::new(1.0, 0.0));
            }
        }
    }
}

#[test]
fn from_scalar() {
    let n = 10;
    let v = Vector::<i8>::from_scalar([n], -42);
    for i in 0..n {
        assert!(v[[i]] == -42);
    }
    let m = Matrix::<i16>::from_scalar([n,n], -42);
    for i in 0..n {
        for j in 0..n {
            assert!(m[[i,j]] == -42);
        }
    }
    let t = TensorN::<i32>::from_scalar([n,n,n], -42);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                assert!(t[[i,j,k]] == -42);
            }
        }
    }
}

#[test]
fn indexing() {
    let n = 3;
    let v = Vector::<u16>::from_slice([n], [0, 1, 2]);
    for i in 0..n {
        assert!(v[[i]] == i as u16);
    }
    let m = Matrix::<f32>::from_slice([n,n], [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2]);
    for i in 0..n {
        for j in 0..n {
            assert!(m[[i,j]] == i as f32 + (j as f32) / 10.0);
        }
    }
    let t = TensorN::<f64>::from_slice([n,n,n], 
            [00.0, 00.1, 00.2, 01.0, 01.1, 01.2, 02.0, 02.1, 02.2, 
             10.0, 10.1, 10.2, 11.0, 11.1, 11.2, 12.0, 12.1, 12.2, 
             20.0, 20.1, 20.2, 21.0, 21.1, 21.2, 22.0, 22.1, 22.2]);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                assert!(t[[i,j,k]] == (i as f64) * 10.0 + j as f64 + (k as f64) / 10.0);
            }
        }
    }
}

#[test]
fn indexing_mut() {
    let n = 3;
    let mut v = Vector::<u8>::zeros([n]);
    for i in 0..n {
        v[[i]] = i as u8;
    }
    for i in 0..n {
        assert!(v[[i]] == i as u8);
    }
    let mut m = Matrix::<f32>::zeros([n,n]);
    for i in 0..n {
        for j in 0..n {
            m[[i,j]] = i as f32 + (j as f32) / 10.0;
        }
    }
    for i in 0..n {
        for j in 0..n {
            assert!(m[[i,j]] == i as f32 + (j as f32) / 10.0);
        }
    }
    let mut t = TensorN::<f64>::zeros([n,n,n]);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                t[[i,j,k]] = (i as f64) * 10.0 + j as f64 + (k as f64) / 10.0;
            }
        }
    }
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                assert!(t[[i,j,k]] == (i as f64) * 10.0 + j as f64 + (k as f64) / 10.0);
            }
        }
    }
}

#[test]
fn cast() {
    let n = 10;
    let v = Vector::<f64>::ones([n]);
    let v2 = Vector::<u16>::from_tensor(&v);
    for i in 0..n {
        assert!(v2[[i]] == 1);
    }
}

#[test]
fn cast_dimensionnality() {
    let n = 10;
    let t = TensorN::<f32>::from_slice([1,n,1], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let m = Matrix::<u8>::from_tensor(&t);
    let v = Vector::<f32>::from_tensor(&m);
    for i in 0..n {
        assert!(v[[i]] == i as f32);
    }
}


