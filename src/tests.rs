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

#[test]
fn vector_insert() {
    let mut v = Vector::<f32>::zeros([0]);
    let v1 = Vector::<f32>::from_slice([3], [6.0, 7.0, 8.0]);
    let v2 = Vector::<f32>::from_slice([3], [0.0, 1.0, 2.0]);
    let v3 = Vector::<f32>::from_slice([3], [3.0, 4.0, 5.0]);
    v.insert(0, 0, &v1);
    v.insert(0, 0, &v2);
    v.insert(0, 3, &v3); 
    for i in 0..9 {
        assert!(v[[i]] == i as f32);
    }
}

#[test]
fn matrix_insert() {
    let mut m = Matrix::<f32>::zeros([3, 0]);
    let m1 = Matrix::<f32>::from_slice([3, 1], [0.0, 1.0, 4.0]);
    let m2 = Matrix::<f32>::from_slice([3, 2], [0.1, 0.4, 1.1, 1.4, 4.1, 4.4]);
    let m3 = Matrix::<f32>::from_slice([2, 3], [2.0, 2.1, 2.4, 3.0, 3.1, 3.4]);
    let m4 = Matrix::<f32>::from_slice([5, 2], [0.2, 0.3, 1.2, 1.3, 2.2, 2.3, 3.2, 3.3, 4.2, 4.3]);
    m.insert(1, 0, &m1); 
    m.insert(1, 1, &m2);
    m.insert(0, 2, &m3);
    m.insert(1, 2, &m4); 
    for i in 0..5 {
        for j in 0..5 {
            assert!(m[[i,j]] == i as f32 + (j as f32) / 10.0);
        }
    }
}

#[test]
fn tensor_insert() {
    let mut t = TensorN::<f32>::zeros([3, 3, 0]);
    let t1 = TensorN::<f32>::from_slice([3, 3, 1], [0.0, 1.0, 4.0, 10.0, 11.0, 14.0, 40.0, 41.0, 44.0]);
    let t2 = TensorN::<f32>::from_slice([3, 3, 2], [0.1, 0.4, 1.1, 1.4, 4.1, 4.4, 10.1, 10.4, 11.1, 11.4, 14.1, 14.4, 40.1, 40.4, 41.1, 41.4, 44.1, 44.4]);
    let t3 = TensorN::<f32>::from_slice([2, 3, 3], [20.0, 20.1, 20.4, 21.0, 21.1, 21.4, 24.0, 24.1, 24.4, 30.0, 30.1, 30.4, 31.0, 31.1, 31.4, 34.0, 34.1, 34.4]);
    let t4 = TensorN::<f32>::from_slice([5, 2, 3], [2.0, 2.1, 2.4, 3.0, 3.1, 3.4, 12.0, 12.1, 12.4, 13.0, 13.1, 13.4, 22.0, 22.1, 22.4, 23.0, 23.1, 23.4, 32.0, 32.1, 32.4, 33.0, 33.1, 33.4, 42.0, 42.1, 42.4, 43.0, 43.1, 43.4]);
    let t5 = TensorN::<f32>::from_slice([5, 5, 2], &[0.2, 0.3, 1.2, 1.3, 2.2, 2.3, 3.2, 3.3, 4.2, 4.3, 10.2, 10.3, 11.2, 11.3, 12.2, 12.3, 13.2, 13.3, 14.2, 14.3, 20.2, 20.3, 21.2, 21.3, 22.2, 22.3, 23.2, 23.3, 24.2, 24.3, 30.2, 30.3, 31.2, 31.3, 32.2, 32.3, 33.2, 33.3, 34.2, 34.3, 40.2, 40.3, 41.2, 41.3, 42.2, 42.3, 43.2, 43.3, 44.2, 44.3][..]);
    t.insert(2, 0, &t1); 
    t.insert(2, 1, &t2); 
    t.insert(0, 2, &t3); 
    t.insert(1, 2, &t4); 
    t.insert(2, 2, &t5); 
    for i in 0..5 {
        for j in 0..5 {
            for k in 0..5 {
                assert!(t[[i,j,k]] == (i as f32) * 10.0 + j as f32 + (k as f32) / 10.0);
            }
        }
    }
}

#[test]
fn vector_extract() {
    let v = Vector::<f64>::from_slice([5], [0.0, 1.0, 2.0, 3.0, 4.0]);   
    assert!(v.extract([0..2]).get_raw_array() == [0.0, 1.0]);
    assert!(v.extract([2..4]).get_raw_array() == [2.0, 3.0]);
    assert!(v.extract([3..5]).get_raw_array() == [3.0, 4.0]);
    assert!(v.extract([0..5]).get_raw_array() == [0.0, 1.0, 2.0, 3.0, 4.0]);
    assert!(v.extract([5..5]).get_raw_array() == []);
}

#[test]
fn matrix_extract() {
    let m = Matrix::<f32>::from_slice([3,3], [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2]);
    assert!(m.extract([0..2,0..2]).get_raw_array() == [0.0, 0.1, 1.0, 1.1]);
    assert!(m.extract([0..2,1..3]).get_raw_array() == [0.1, 0.2, 1.1, 1.2]);
    assert!(m.extract([1..3,0..2]).get_raw_array() == [1.0, 1.1, 2.0, 2.1]);
    assert!(m.extract([1..3,1..3]).get_raw_array() == [1.1, 1.2, 2.1, 2.2]);
    assert!(m.extract([0..3,0..3]).get_raw_array() == [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2]);
    assert!(m.extract([0..0,0..3]).get_raw_array() == []);
    assert!(m.extract([0..3,0..0]).get_raw_array() == []);
}

#[test]
fn tensor_extract() {
    let t = TensorN::<f64>::from_slice([3,3,3], 
        [00.0, 00.1, 00.2, 01.0, 01.1, 01.2, 02.0, 02.1, 02.2, 
         10.0, 10.1, 10.2, 11.0, 11.1, 11.2, 12.0, 12.1, 12.2, 
         20.0, 20.1, 20.2, 21.0, 21.1, 21.2, 22.0, 22.1, 22.2]);

    assert!(t.extract([0..2,0..2,0..2]).get_raw_array() == [0.0, 0.1, 1.0, 1.1, 10.0, 10.1, 11.0, 11.1]);
    assert!(t.extract([0..2,0..2,1..3]).get_raw_array() == [0.1, 0.2, 1.1, 1.2, 10.1, 10.2, 11.1, 11.2]);
    assert!(t.extract([0..2,1..3,0..2]).get_raw_array() == [1.0, 1.1, 2.0, 2.1, 11.0, 11.1, 12.0, 12.1]);
    assert!(t.extract([0..2,1..3,1..3]).get_raw_array() == [1.1, 1.2, 2.1, 2.2, 11.1, 11.2, 12.1, 12.2]);
    assert!(t.extract([1..3,0..2,0..2]).get_raw_array() == [10.0, 10.1, 11.0, 11.1, 20.0, 20.1, 21.0, 21.1]);
    assert!(t.extract([1..3,0..2,1..3]).get_raw_array() == [10.1, 10.2, 11.1, 11.2, 20.1, 20.2, 21.1, 21.2]);
    assert!(t.extract([1..3,1..3,0..2]).get_raw_array() == [11.0, 11.1, 12.0, 12.1, 21.0, 21.1, 22.0, 22.1]);
    assert!(t.extract([1..3,1..3,1..3]).get_raw_array() == [11.1, 11.2, 12.1, 12.2, 21.1, 21.2, 22.1, 22.2]);
    assert!(t.extract([0..3,0..3,0..3]).get_raw_array() == [00.0, 00.1, 00.2, 01.0, 01.1, 01.2, 02.0, 02.1, 02.2, 
                                                            10.0, 10.1, 10.2, 11.0, 11.1, 11.2, 12.0, 12.1, 12.2, 
                                                            20.0, 20.1, 20.2, 21.0, 21.1, 21.2, 22.0, 22.1, 22.2]);
    assert!(t.extract([0..3,0..3,0..0]).get_raw_array() == []);
    assert!(t.extract([0..3,0..0,0..3]).get_raw_array() == []);
    assert!(t.extract([0..0,0..3,0..3]).get_raw_array() == []);
    assert!(t.extract([0..3,0..3,3..3]).get_raw_array() == []);
    assert!(t.extract([0..3,3..3,0..3]).get_raw_array() == []);
    assert!(t.extract([3..3,0..3,0..3]).get_raw_array() == []);
}
