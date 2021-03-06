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

#[test]
fn vector_remove() {
    let mut v = Vector::<f64>::from_slice([5], [0.0, 1.0, 2.0, 3.0, 4.0]);   
    v.remove([0..1]);
    assert!(v.get_raw_array() == [1.0, 2.0, 3.0, 4.0]);
    v.remove([3..4]);
    assert!(v.get_raw_array() == [1.0, 2.0, 3.0]);
    v.remove([1..2]);
    assert!(v.get_raw_array() == [1.0, 3.0]);
}

#[test]
fn matrix_remove() {
    let mut m = Matrix::<f64>::from_slice([5,5], [
        0.0, 0.1, 0.2, 0.3, 0.4,
        1.0, 1.1, 1.2, 1.3, 1.4,
        2.0, 2.1, 2.2, 2.3, 2.4,
        3.0, 3.1, 3.2, 3.3, 3.4,
        4.0, 4.1, 4.2, 4.3, 4.4,
    ]);   
    m.remove([1..3,0..5]);
    assert!(m.get_raw_array() == [
        0.0, 0.1, 0.2, 0.3, 0.4,
        3.0, 3.1, 3.2, 3.3, 3.4,
        4.0, 4.1, 4.2, 4.3, 4.4,
    ]);
    m.remove([0..3,2..4]);
    assert!(m.get_raw_array() == [
        0.0, 0.1, 0.4,
        3.0, 3.1, 3.4,
        4.0, 4.1, 4.4,
    ]);
    m.remove([2..3,0..3]);
    assert!(m.get_raw_array() == [
        0.0, 0.1, 0.4,
        3.0, 3.1, 3.4,
    ]);
    m.remove([0..2,2..3]);
    assert!(m.get_raw_array() == [
        0.0, 0.1,
        3.0, 3.1,
    ]);
}

#[test]
fn tensor_remove() {
    let data = [
        00.0, 00.1, 00.2, 00.3,
        01.0, 01.1, 01.2, 01.3,
        02.0, 02.1, 02.2, 02.3,
        03.0, 03.1, 03.2, 03.3,

        10.0, 10.1, 10.2, 10.3,
        11.0, 11.1, 11.2, 11.3,
        12.0, 12.1, 12.2, 12.3,
        13.0, 13.1, 13.2, 13.3,

        20.0, 20.1, 20.2, 20.3,
        21.0, 21.1, 21.2, 21.3,
        22.0, 22.1, 22.2, 22.3,
        23.0, 23.1, 23.2, 23.3,

        30.0, 30.1, 30.2, 30.3,
        31.0, 31.1, 31.2, 31.3,
        32.0, 32.1, 32.2, 32.3,
        33.0, 33.1, 33.2, 33.3,
    ];
    let mut t = TensorN::<f64>::from_slice([4,4,4], &data[..]);
    t.remove([0..2,0..4,0..4]);
    assert!(t.get_raw_array() == [
        20.0, 20.1, 20.2, 20.3,
        21.0, 21.1, 21.2, 21.3,
        22.0, 22.1, 22.2, 22.3,
        23.0, 23.1, 23.2, 23.3,

        30.0, 30.1, 30.2, 30.3,
        31.0, 31.1, 31.2, 31.3,
        32.0, 32.1, 32.2, 32.3,
        33.0, 33.1, 33.2, 33.3,
    ]);
    t.remove([0..2,0..2,0..4]);
    assert!(t.get_raw_array() == [
        22.0, 22.1, 22.2, 22.3,
        23.0, 23.1, 23.2, 23.3,

        32.0, 32.1, 32.2, 32.3,
        33.0, 33.1, 33.2, 33.3,
    ]);
    t.remove([0..2,0..2,0..2]);
    assert!(t.get_raw_array() == [
        22.2, 22.3,
        23.2, 23.3,

        32.2, 32.3,
        33.2, 33.3,
    ]);
    t.remove([1..2,0..2,0..2]);
    assert!(t.get_raw_array() == [
        22.2, 22.3,
        23.2, 23.3,
    ]);
    t.remove([0..1,1..2,0..2]);
    assert!(t.get_raw_array() == [
        22.2, 22.3,
    ]);
    t.remove([0..1,0..1,1..2]);
    assert!(t.get_raw_array() == [
        22.2,
    ]);
}

#[test]
fn vector_split() {
    let mut v = Vector::<u16>::from_slice([6], [0, 1, 2, 3, 4, 5]);
    assert!(v.right_split(0, 4).get_raw_array() == [4, 5]);
    assert!(v.get_raw_array() == [0, 1, 2, 3]);
    assert!(v.left_split(0, 2).get_raw_array() == [0, 1]);
    assert!(v.get_raw_array() == [2, 3]);
}

#[test]
fn matrix_split() {
    let mut m = Matrix::<f32>::from_slice([4,4], [
        0.0, 0.1, 0.2, 0.3,
        1.0, 1.1, 1.2, 1.3,
        2.0, 2.1, 2.2, 2.3,
        3.0, 3.1, 3.2, 3.3,
    ]);

    assert!(m.right_split(1, 2).get_raw_array() == [0.2, 0.3, 1.2, 1.3, 2.2, 2.3, 3.2, 3.3]);
    assert!(m.get_raw_array() == [0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1]);
    assert!(m.left_split(0, 2).get_raw_array() == [0.0, 0.1, 1.0, 1.1]);
    assert!(m.get_raw_array() == [2.0, 2.1, 3.0, 3.1]);
    assert!(m.right_split(1, 1).get_raw_array() == [2.1, 3.1]);
    assert!(m.get_raw_array() == [2.0, 3.0]);
    assert!(m.left_split(0, 1).get_raw_array() == [2.0]);
    assert!(m.get_raw_array() == [3.0]);
}

#[test]
fn tensor_split() {
    let data = [
        00.0, 00.1, 00.2, 00.3,
        01.0, 01.1, 01.2, 01.3,
        02.0, 02.1, 02.2, 02.3,
        03.0, 03.1, 03.2, 03.3,

        10.0, 10.1, 10.2, 10.3,
        11.0, 11.1, 11.2, 11.3,
        12.0, 12.1, 12.2, 12.3,
        13.0, 13.1, 13.2, 13.3,

        20.0, 20.1, 20.2, 20.3,
        21.0, 21.1, 21.2, 21.3,
        22.0, 22.1, 22.2, 22.3,
        23.0, 23.1, 23.2, 23.3,

        30.0, 30.1, 30.2, 30.3,
        31.0, 31.1, 31.2, 31.3,
        32.0, 32.1, 32.2, 32.3,
        33.0, 33.1, 33.2, 33.3,
    ];
    let mut t = TensorN::<f64>::from_slice([4,4,4], &data[..]);
    assert!(t.left_split(2, 2).get_raw_array() == [
        00.0, 00.1,
        01.0, 01.1,
        02.0, 02.1,
        03.0, 03.1,

        10.0, 10.1,
        11.0, 11.1,
        12.0, 12.1,
        13.0, 13.1,

        20.0, 20.1,
        21.0, 21.1,
        22.0, 22.1,
        23.0, 23.1,

        30.0, 30.1,
        31.0, 31.1,
        32.0, 32.1,
        33.0, 33.1,
    ]);
    assert!(t.get_raw_array() == [
        00.2, 00.3,
        01.2, 01.3,
        02.2, 02.3,
        03.2, 03.3,

        10.2, 10.3,
        11.2, 11.3,
        12.2, 12.3,
        13.2, 13.3,

        20.2, 20.3,
        21.2, 21.3,
        22.2, 22.3,
        23.2, 23.3,

        30.2, 30.3,
        31.2, 31.3,
        32.2, 32.3,
        33.2, 33.3,
    ]);
    assert!(t.right_split(1, 2).get_raw_array() == [
        02.2, 02.3,
        03.2, 03.3,

        12.2, 12.3,
        13.2, 13.3,

        22.2, 22.3,
        23.2, 23.3,

        32.2, 32.3,
        33.2, 33.3,
    ]);
    assert!(t.get_raw_array() == [
        00.2, 00.3,
        01.2, 01.3,

        10.2, 10.3,
        11.2, 11.3,

        20.2, 20.3,
        21.2, 21.3,

        30.2, 30.3,
        31.2, 31.3,
    ]);
    assert!(t.left_split(0, 2).get_raw_array() == [
        00.2, 00.3,
        01.2, 01.3,

        10.2, 10.3,
        11.2, 11.3,
    ]);
    assert!(t.get_raw_array() == [
        20.2, 20.3,
        21.2, 21.3,

        30.2, 30.3,
        31.2, 31.3,
    ]);
}

#[test]
fn matrix_transpose() {
    let mut m = Matrix::<u64>::from_slice([3,4], [
        00, 01, 02, 03,
        10, 11, 12, 13,
        20, 21, 22, 23,
    ]);
    m.transpose();
    assert!(m.shape() == [4,3]);
    assert!(m.get_raw_array() == [
        00, 10, 20,
        01, 11, 21,
        02, 12, 22,
        03, 13, 23,
    ]);
    m.transpose();
    assert!(m.shape() == [3,4]);
    assert!(m.get_raw_array() == [
        00, 01, 02, 03,
        10, 11, 12, 13,
        20, 21, 22, 23,
    ]);
    let mut m = Matrix::<u64>::from_slice([4,5], [
        00, 01, 02, 03, 04,
        10, 11, 12, 13, 14,
        20, 21, 22, 23, 24,
        30, 31, 32, 33, 34,
    ]);
    m.transpose();
    assert!(m.shape() == [5,4]);
    assert!(m.get_raw_array() == [
        00, 10, 20, 30,
        01, 11, 21, 31,
        02, 12, 22, 32,
        03, 13, 23, 33,
        04, 14, 24, 34,
    ]);
    m.transpose();
    assert!(m.shape() == [4,5]);
    assert!(m.get_raw_array() == [
        00, 01, 02, 03, 04,
        10, 11, 12, 13, 14,
        20, 21, 22, 23, 24,
        30, 31, 32, 33, 34,
    ]);
    let mut m = Matrix::<u64>::from_slice([5,6], [
        00, 01, 02, 03, 04, 05,
        10, 11, 12, 13, 14, 15,
        20, 21, 22, 23, 24, 25,
        30, 31, 32, 33, 34, 35,
        40, 41, 42, 43, 44, 45,
    ]);
    m.transpose();
    assert!(m.shape() == [6,5]);
    assert!(m.get_raw_array() == [
        00, 10, 20, 30, 40,
        01, 11, 21, 31, 41,
        02, 12, 22, 32, 42,
        03, 13, 23, 33, 43,
        04, 14, 24, 34, 44,
        05, 15, 25, 35, 45,
    ]);
    m.transpose();
    assert!(m.shape() == [5,6]);
    assert!(m.get_raw_array() == [
        00, 01, 02, 03, 04, 05,
        10, 11, 12, 13, 14, 15,
        20, 21, 22, 23, 24, 25,
        30, 31, 32, 33, 34, 35,
        40, 41, 42, 43, 44, 45,
    ]);
}

#[test]
fn tensor_transpose() {
    let data = [
        000, 001, 002, 003, 004,
        010, 011, 012, 013, 014,
        020, 021, 022, 023, 024,
        030, 031, 032, 033, 034,

        100, 101, 102, 103, 104,
        110, 111, 112, 113, 114,
        120, 121, 122, 123, 124,
        130, 131, 132, 133, 134,

        200, 201, 202, 203, 204,
        210, 211, 212, 213, 214,
        220, 221, 222, 223, 224,
        230, 231, 232, 233, 234,
    ];
    let mut t = TensorN::<u32>::from_slice([3,4,5], &data[..]);
    t.transpose();
    assert!(t.shape() == [5,4,3]);
    assert!(t.get_raw_array() == &[
        000, 100, 200,
        010, 110, 210,
        020, 120, 220,
        030, 130, 230,

        001, 101, 201,
        011, 111, 211,
        021, 121, 221,
        031, 131, 231,

        002, 102, 202,
        012, 112, 212,
        022, 122, 222,
        032, 132, 232,

        003, 103, 203,
        013, 113, 213,
        023, 123, 223,
        033, 133, 233,

        004, 104, 204,
        014, 114, 214,
        024, 124, 224,
        034, 134, 234,

    ][..]);
    t.transpose();
    assert!(t.shape() == [3,4,5]);
    assert!(t.get_raw_array() == &data[..]);
    let mut t = TensorN::<u16>::from_slice([2,3,4], [
        000, 001, 002, 003,
        010, 011, 012, 013,
        020, 021, 022, 023,

        100, 101, 102, 103,
        110, 111, 112, 113,
        120, 121, 122, 123,
    ]);
    t.transpose();
    assert!(t.shape() == [4,3,2]);
    assert!(t.get_raw_array() == [
        000, 100,
        010, 110,
        020, 120,

        001, 101,
        011, 111,
        021, 121,

        002, 102,
        012, 112,
        022, 122,

        003, 103,
        013, 113,
        023, 123,
    ]);
    t.transpose();
    assert!(t.shape() == [2,3,4]);
    assert!(t.get_raw_array() == [
        000, 001, 002, 003,
        010, 011, 012, 013,
        020, 021, 022, 023,

        100, 101, 102, 103,
        110, 111, 112, 113,
        120, 121, 122, 123,
    ]);
}

#[test]
fn vector_assign() {
    let mut v = Vector::<i16>::zeros([10]);
    let v1 = Vector::<i16>::from_slice([3], [0, 1, 2]);
    let v2 = Vector::<i16>::from_slice([4], [3, 4, 5, 6]);
    let v3 = Vector::<i16>::from_slice([3], [7, 8, 9]);

    v.assign([3..7], &v2);
    v.assign([0..3], &v1);
    v.assign([7..10], &v3);
    
    assert!(v.get_raw_array() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

#[test]
fn matrix_assign() {
    let mut m = Matrix::<i32>::zeros([3,3]);
    let m1 = Matrix::<i32>::from_slice([1,3], [00, 01, 02]);
    let m2 = Matrix::<i32>::from_slice([3,1], [00, 10, 20]);
    let m3 = Matrix::<i32>::from_slice([2,2], [11, 12, 21, 22]);

    m.assign([0..1,0..3], &m1);
    m.assign([0..3,0..1], &m2);
    m.assign([1..3,1..3], &m3);
    
    assert!(m.get_raw_array() == [00, 01, 02, 10, 11, 12, 20, 21, 22]);
}

#[test]
fn tensor_assign() {
    let mut t = TensorN::<i64>::zeros([3,3,3]);
    let t1 = TensorN::<i64>::from_slice([1,3,3], [000, 001, 002, 010, 011, 012, 020, 021, 022]);
    let t2 = TensorN::<i64>::from_slice([3,3,1], [000, 010, 020, 100, 110, 120, 200, 210, 220]);
    let t3 = TensorN::<i64>::from_slice([3,1,3], [000, 001, 002, 100, 101, 102, 200, 201, 202]);
    let t4 = TensorN::<i64>::from_slice([2,2,2], [111, 112, 121, 122, 211, 212, 221, 222]);

    t.assign([0..1,0..3,0..3], &t1);
    t.assign([0..3,0..3,0..1], &t2);
    t.assign([0..3,0..1,0..3], &t3);
    t.assign([1..3,1..3,1..3], &t4);
    
    assert!(t.get_raw_array() == [000, 001, 002, 010, 011, 012, 020, 021, 022, 100, 101, 102, 110, 111, 112, 120, 121, 122, 200, 201, 202, 210, 211, 212, 220, 221, 222]);
}
