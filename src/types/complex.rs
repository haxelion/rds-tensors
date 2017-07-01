use std::cmp::{Eq,PartialEq};
use std::f32;
use std::f64;
use std::fmt;
use std::ops::{Add,AddAssign,Div,DivAssign,Mul,MulAssign,Neg,Sub,SubAssign};
use std::num::ParseFloatError;
use std::str::FromStr;

/// Trait defining basic operations on complex types.
pub trait Complex<T> where Self : Sized + Copy {

    /// Create a complex from a set of cartesian coordinates.
    fn cartesian(re : T, im : T) -> Self;
        
    /// Create a complex from a set of polar coordinates.
    fn polar(abs : T, arg : T) -> Self;

    /// Return the real part of the complex.
    fn re(self) -> T; 

    /// Return the imaginary part of the complex.
    fn im(self) -> T; 

    /// Return the absolute value of the complex.
    fn abs(self) -> T;

    /// Return the argument of the complex.
    fn arg(self) -> T; 

    /// Return a tuple forming the polar coordinates of the complex (abs,arg).
    fn to_polar(self) -> (T,T)  {
        (self.abs(), self.arg())
    }

    /// Return the conjugate value of the complex.
    fn conj(self) -> Self;

    /// Return the reciprocal value of the complex.
    fn reciprocal(self) -> Self;
}

#[repr(C)]
#[derive(Clone,Copy,Debug)]
pub struct c32 {
    pub re : f32,
    pub im : f32
}

#[repr(C)]
#[derive(Clone,Copy,Debug)]
pub struct c64 {
    pub re : f64,
    pub im : f64
}

// ==================== c32 ====================

impl c32 {
    pub fn new(re : f32, im : f32) -> c32 {
        c32 {
            re : re,
            im : im,
        }
    }
}

impl Complex<f32> for c32 {

    fn cartesian(re : f32, im : f32) -> c32{
        c32::new(re, im)
    }

    fn polar(arg : f32, abs : f32) -> c32 {
        c32::new(arg * abs.cos(), arg * abs.sin())
    }
    
    fn re(self) -> f32 {
        self.re
    }

    fn im(self) -> f32 {
        self.im
    }

    fn abs(self) -> f32 {
        return (self.re * self.re + self.im * self.im).sqrt();
    }

    fn arg(self) -> f32 {
        self.im.atan2(self.re)
    }

    fn conj(self) -> c32 {
        c32::new(self.re, -self.im)
    }

    fn reciprocal(self) -> c32 {
        let base = self.re * self.re + self.im * self.im;
        c32::new(self.re / base, -self.im / base)
    }
}

impl Add for c32 {
    type Output = c32;

    fn add(self, rhs: c32) -> c32 {
        c32::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl AddAssign for c32 {

    fn add_assign(&mut self, rhs: c32) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Div for c32 {
    type Output = c32;

    fn div(self, rhs: c32) -> c32 {
        self * rhs.reciprocal()
    }
}

impl DivAssign for c32 {

    fn div_assign(&mut self, rhs: c32) {
        *self *= rhs.reciprocal();
    }
}

impl Mul for c32 {
    type Output = c32;

    fn mul(self, rhs: c32) -> c32 {
        let re = self.re * rhs.re - self.im * rhs.im;
        let im = self.re * rhs.im + self.im * rhs.re;
        c32::new(re, im)
    }
}

impl MulAssign for c32 {

    fn mul_assign(&mut self, rhs: c32) {
        let re = self.re * rhs.re - self.im * rhs.im;
        let im = self.re * rhs.im + self.im * rhs.re;
        self.re = re;
        self.im = im; 
    }
}

impl Neg for c32 {
    type Output = c32;

    fn neg(self) -> c32 {
        c32::new(-self.re, -self.im)
    }
}

impl Sub for c32 {
    type Output = c32;

    fn sub(self, rhs: c32) -> c32 {
        c32::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl SubAssign for c32 {

    fn sub_assign(&mut self, rhs: c32) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl PartialEq<c32> for c32 {

    fn eq(&self, rhs : &c32) -> bool {
        self.re == rhs.re && self.im == rhs.im
    }
}

impl Eq for c32 {
}

impl FromStr for c32 {
    type Err = ParseFloatError;

    fn from_str(s: &str) -> Result<c32, Self::Err> {
        let mut v = c32::new(0.0, 0.0);
        if s.ends_with('i') || s.ends_with('j') {
            let imend = s.len()-1;
            let mut imstart = 0usize;

            if let Some(p) = s.rfind('-') {
                if p > 0 {
                    imstart = p;
                }
            }
            if let Some(p) = s.rfind('+') {
                if p > 0 {
                    imstart = p;
                }
            }

            if imstart > 0 {
                v.re = match f32::from_str(&s[0..imstart]) {
                    Ok(re) => re,
                    Err(e) => {
                        return Err(e);
                    }
                };
            }
            v.im = match f32::from_str(&s[imstart..imend]) {
                Ok(im) => im,
                Err(e) => {
                    return Err(e);
                }
            };
        }
        // No imaginary part
        else {
            v.re = match f32::from_str(s) {
                Ok(re) => re,
                Err(e) => {
                    return Err(e);
                }
            };
        }
        return Ok(v);
    }
}

impl fmt::Display for c32 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{:+}j", self.re, self.im)
    }
}

// ==================== c64 ====================

impl c64 {
    pub fn new(re : f64, im : f64) -> c64 {
        c64 {
            re : re,
            im : im,
        }
    }
}

impl Complex<f64> for c64 {

    fn cartesian(re : f64, im : f64) -> c64{
        c64::new(re, im)
    }

    fn polar(arg : f64, abs : f64) -> c64{
        c64::new(arg * abs.cos(), arg * abs.sin())
    }
    
    fn re(self) -> f64 {
        self.re
    }

    fn im(self) -> f64 {
        self.im
    }

    fn abs(self) -> f64 {
        return (self.re * self.re + self.im * self.im).sqrt();
    }

    fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    fn conj(self) -> c64 {
        c64::new(self.re, -self.im)
    }

    fn reciprocal(self) -> c64 {
        let base = self.re * self.re + self.im * self.im;
        c64::new(self.re / base, -self.im / base)
    }
}

impl Add for c64 {
    type Output = c64;

    fn add(self, rhs: c64) -> c64 {
        c64::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl AddAssign for c64 {

    fn add_assign(&mut self, rhs: c64) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Div for c64 {
    type Output = c64;

    fn div(self, rhs: c64) -> c64 {
        self * rhs.reciprocal()
    }
}

impl DivAssign for c64 {

    fn div_assign(&mut self, rhs: c64) {
        *self *= rhs.reciprocal();
    }
}

impl Mul for c64 {
    type Output = c64;

    fn mul(self, rhs: c64) -> c64 {
        let re = self.re * rhs.re - self.im * rhs.im;
        let im = self.re * rhs.im + self.im * rhs.re;
        c64::new(re, im)
    }
}

impl MulAssign for c64 {

    fn mul_assign(&mut self, rhs: c64) {
        let re = self.re * rhs.re - self.im * rhs.im;
        let im = self.re * rhs.im + self.im * rhs.re;
        self.re = re;
        self.im = im; 
    }
}

impl Neg for c64 {
    type Output = c64;

    fn neg(self) -> c64 {
        c64::new(-self.re, -self.im)
    }
}

impl Sub for c64 {
    type Output = c64;

    fn sub(self, rhs: c64) -> c64 {
        c64::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl SubAssign for c64 {

    fn sub_assign(&mut self, rhs: c64) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl PartialEq<c64> for c64 {

    fn eq(&self, rhs : &c64) -> bool {
        self.re == rhs.re && self.im == rhs.im
    }
}

impl Eq for c64 {
}

impl FromStr for c64 {
    type Err = ParseFloatError;

    fn from_str(s: &str) -> Result<c64, Self::Err> {
        let mut v = c64::new(0.0, 0.0);
        if s.ends_with('i') || s.ends_with('j') {
            let imend = s.len()-1;
            let mut imstart = 0usize;

            if let Some(p) = s.rfind('-') {
                if p > 0 {
                    imstart = p;
                }
            }
            if let Some(p) = s.rfind('+') {
                if p > 0 {
                    imstart = p;
                }
            }

            if imstart > 0 {
                v.re = match f64::from_str(&s[0..imstart]) {
                    Ok(re) => re,
                    Err(e) => {
                        return Err(e);
                    }
                };
            }
            v.im = match f64::from_str(&s[imstart..imend]) {
                Ok(im) => im,
                Err(e) => {
                    return Err(e);
                }
            };
        }
        // No imaginary part
        else {
            v.re = match f64::from_str(s) {
                Ok(re) => re,
                Err(e) => {
                    return Err(e);
                }
            };
        }
        return Ok(v);
    }
}

impl fmt::Display for c64 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{:+}j", self.re, self.im)
    }
}
