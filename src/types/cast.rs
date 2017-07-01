use types::complex::{c32,c64};

/// Trait allowing to cast Self into T.
pub trait CastTo<T : Copy> {
    fn cast_to(self) -> T;
}

impl CastTo<u8>  for u8  { fn cast_to(self) -> u8  { self as u8   } }
impl CastTo<u16> for u8  { fn cast_to(self) -> u16 { self as u16  } }
impl CastTo<u32> for u8  { fn cast_to(self) -> u32 { self as u32  } }
impl CastTo<u64> for u8  { fn cast_to(self) -> u64 { self as u64  } }
impl CastTo<i8>  for u8  { fn cast_to(self) -> i8  { self as i8   } }
impl CastTo<i16> for u8  { fn cast_to(self) -> i16 { self as i16  } }
impl CastTo<i32> for u8  { fn cast_to(self) -> i32 { self as i32  } }
impl CastTo<i64> for u8  { fn cast_to(self) -> i64 { self as i64  } }
impl CastTo<f32> for u8  { fn cast_to(self) -> f32 { self as f32  } }
impl CastTo<f64> for u8  { fn cast_to(self) -> f64 { self as f64  } }
impl CastTo<c32> for u8  { fn cast_to(self) -> c32 { c32::new(self as f32, 0.0) } }
impl CastTo<c64> for u8  { fn cast_to(self) -> c64 { c64::new(self as f64, 0.0) } }

impl CastTo<u8>  for u16 { fn cast_to(self) -> u8  { self as u8   } }
impl CastTo<u16> for u16 { fn cast_to(self) -> u16 { self as u16  } }
impl CastTo<u32> for u16 { fn cast_to(self) -> u32 { self as u32  } }
impl CastTo<u64> for u16 { fn cast_to(self) -> u64 { self as u64  } }
impl CastTo<i8>  for u16 { fn cast_to(self) -> i8  { self as i8   } }
impl CastTo<i16> for u16 { fn cast_to(self) -> i16 { self as i16  } }
impl CastTo<i32> for u16 { fn cast_to(self) -> i32 { self as i32  } }
impl CastTo<i64> for u16 { fn cast_to(self) -> i64 { self as i64  } }
impl CastTo<f32> for u16 { fn cast_to(self) -> f32 { self as f32  } }
impl CastTo<f64> for u16 { fn cast_to(self) -> f64 { self as f64  } }
impl CastTo<c32> for u16 { fn cast_to(self) -> c32 { c32::new(self as f32, 0.0) } }
impl CastTo<c64> for u16 { fn cast_to(self) -> c64 { c64::new(self as f64, 0.0) } }

impl CastTo<u8>  for u32 { fn cast_to(self) -> u8  { self as u8   } }
impl CastTo<u16> for u32 { fn cast_to(self) -> u16 { self as u16  } }
impl CastTo<u32> for u32 { fn cast_to(self) -> u32 { self as u32  } }
impl CastTo<u64> for u32 { fn cast_to(self) -> u64 { self as u64  } }
impl CastTo<i8>  for u32 { fn cast_to(self) -> i8  { self as i8   } }
impl CastTo<i16> for u32 { fn cast_to(self) -> i16 { self as i16  } }
impl CastTo<i32> for u32 { fn cast_to(self) -> i32 { self as i32  } }
impl CastTo<i64> for u32 { fn cast_to(self) -> i64 { self as i64  } }
impl CastTo<f32> for u32 { fn cast_to(self) -> f32 { self as f32  } }
impl CastTo<f64> for u32 { fn cast_to(self) -> f64 { self as f64  } }
impl CastTo<c32> for u32 { fn cast_to(self) -> c32 { c32::new(self as f32, 0.0) } }
impl CastTo<c64> for u32 { fn cast_to(self) -> c64 { c64::new(self as f64, 0.0) } }

impl CastTo<u8>  for u64 { fn cast_to(self) -> u8  { self as u8   } }
impl CastTo<u16> for u64 { fn cast_to(self) -> u16 { self as u16  } }
impl CastTo<u32> for u64 { fn cast_to(self) -> u32 { self as u32  } }
impl CastTo<u64> for u64 { fn cast_to(self) -> u64 { self as u64  } }
impl CastTo<i8>  for u64 { fn cast_to(self) -> i8  { self as i8   } }
impl CastTo<i16> for u64 { fn cast_to(self) -> i16 { self as i16  } }
impl CastTo<i32> for u64 { fn cast_to(self) -> i32 { self as i32  } }
impl CastTo<i64> for u64 { fn cast_to(self) -> i64 { self as i64  } }
impl CastTo<f32> for u64 { fn cast_to(self) -> f32 { self as f32  } }
impl CastTo<f64> for u64 { fn cast_to(self) -> f64 { self as f64  } }
impl CastTo<c32> for u64 { fn cast_to(self) -> c32 { c32::new(self as f32, 0.0) } }
impl CastTo<c64> for u64 { fn cast_to(self) -> c64 { c64::new(self as f64, 0.0) } }

impl CastTo<u8>  for i8  { fn cast_to(self) -> u8  { self as u8   } }
impl CastTo<u16> for i8  { fn cast_to(self) -> u16 { self as u16  } }
impl CastTo<u32> for i8  { fn cast_to(self) -> u32 { self as u32  } }
impl CastTo<u64> for i8  { fn cast_to(self) -> u64 { self as u64  } }
impl CastTo<i8>  for i8  { fn cast_to(self) -> i8  { self as i8   } }
impl CastTo<i16> for i8  { fn cast_to(self) -> i16 { self as i16  } }
impl CastTo<i32> for i8  { fn cast_to(self) -> i32 { self as i32  } }
impl CastTo<i64> for i8  { fn cast_to(self) -> i64 { self as i64  } }
impl CastTo<f32> for i8  { fn cast_to(self) -> f32 { self as f32  } }
impl CastTo<f64> for i8  { fn cast_to(self) -> f64 { self as f64  } }
impl CastTo<c32> for i8  { fn cast_to(self) -> c32 { c32::new(self as f32, 0.0) } }
impl CastTo<c64> for i8  { fn cast_to(self) -> c64 { c64::new(self as f64, 0.0) } }

impl CastTo<u8>  for i16 { fn cast_to(self) -> u8  { self as u8   } }
impl CastTo<u16> for i16 { fn cast_to(self) -> u16 { self as u16  } }
impl CastTo<u32> for i16 { fn cast_to(self) -> u32 { self as u32  } }
impl CastTo<u64> for i16 { fn cast_to(self) -> u64 { self as u64  } }
impl CastTo<i8>  for i16 { fn cast_to(self) -> i8  { self as i8   } }
impl CastTo<i16> for i16 { fn cast_to(self) -> i16 { self as i16  } }
impl CastTo<i32> for i16 { fn cast_to(self) -> i32 { self as i32  } }
impl CastTo<i64> for i16 { fn cast_to(self) -> i64 { self as i64  } }
impl CastTo<f32> for i16 { fn cast_to(self) -> f32 { self as f32  } }
impl CastTo<f64> for i16 { fn cast_to(self) -> f64 { self as f64  } }
impl CastTo<c32> for i16 { fn cast_to(self) -> c32 { c32::new(self as f32, 0.0) } }
impl CastTo<c64> for i16 { fn cast_to(self) -> c64 { c64::new(self as f64, 0.0) } }

impl CastTo<u8>  for i32 { fn cast_to(self) -> u8  { self as u8   } }
impl CastTo<u16> for i32 { fn cast_to(self) -> u16 { self as u16  } }
impl CastTo<u32> for i32 { fn cast_to(self) -> u32 { self as u32  } }
impl CastTo<u64> for i32 { fn cast_to(self) -> u64 { self as u64  } }
impl CastTo<i8>  for i32 { fn cast_to(self) -> i8  { self as i8   } }
impl CastTo<i16> for i32 { fn cast_to(self) -> i16 { self as i16  } }
impl CastTo<i32> for i32 { fn cast_to(self) -> i32 { self as i32  } }
impl CastTo<i64> for i32 { fn cast_to(self) -> i64 { self as i64  } }
impl CastTo<f32> for i32 { fn cast_to(self) -> f32 { self as f32  } }
impl CastTo<f64> for i32 { fn cast_to(self) -> f64 { self as f64  } }
impl CastTo<c32> for i32 { fn cast_to(self) -> c32 { c32::new(self as f32, 0.0) } }
impl CastTo<c64> for i32 { fn cast_to(self) -> c64 { c64::new(self as f64, 0.0) } }

impl CastTo<u8>  for i64 { fn cast_to(self) -> u8  { self as u8   } }
impl CastTo<u16> for i64 { fn cast_to(self) -> u16 { self as u16  } }
impl CastTo<u32> for i64 { fn cast_to(self) -> u32 { self as u32  } }
impl CastTo<u64> for i64 { fn cast_to(self) -> u64 { self as u64  } }
impl CastTo<i8>  for i64 { fn cast_to(self) -> i8  { self as i8   } }
impl CastTo<i16> for i64 { fn cast_to(self) -> i16 { self as i16  } }
impl CastTo<i32> for i64 { fn cast_to(self) -> i32 { self as i32  } }
impl CastTo<i64> for i64 { fn cast_to(self) -> i64 { self as i64  } }
impl CastTo<f32> for i64 { fn cast_to(self) -> f32 { self as f32  } }
impl CastTo<f64> for i64 { fn cast_to(self) -> f64 { self as f64  } }
impl CastTo<c32> for i64 { fn cast_to(self) -> c32 { c32::new(self as f32, 0.0) } }
impl CastTo<c64> for i64 { fn cast_to(self) -> c64 { c64::new(self as f64, 0.0) } }

impl CastTo<u8>  for f32 { fn cast_to(self) -> u8  { self as u8   } }
impl CastTo<u16> for f32 { fn cast_to(self) -> u16 { self as u16  } }
impl CastTo<u32> for f32 { fn cast_to(self) -> u32 { self as u32  } }
impl CastTo<u64> for f32 { fn cast_to(self) -> u64 { self as u64  } }
impl CastTo<i8>  for f32 { fn cast_to(self) -> i8  { self as i8   } }
impl CastTo<i16> for f32 { fn cast_to(self) -> i16 { self as i16  } }
impl CastTo<i32> for f32 { fn cast_to(self) -> i32 { self as i32  } }
impl CastTo<i64> for f32 { fn cast_to(self) -> i64 { self as i64  } }
impl CastTo<f32> for f32 { fn cast_to(self) -> f32 { self as f32  } }
impl CastTo<f64> for f32 { fn cast_to(self) -> f64 { self as f64  } }
impl CastTo<c32> for f32 { fn cast_to(self) -> c32 { c32::new(self as f32, 0.0) } }
impl CastTo<c64> for f32 { fn cast_to(self) -> c64 { c64::new(self as f64, 0.0) } }

impl CastTo<u8>  for f64 { fn cast_to(self) -> u8  { self as u8   } }
impl CastTo<u16> for f64 { fn cast_to(self) -> u16 { self as u16  } }
impl CastTo<u32> for f64 { fn cast_to(self) -> u32 { self as u32  } }
impl CastTo<u64> for f64 { fn cast_to(self) -> u64 { self as u64  } }
impl CastTo<i8>  for f64 { fn cast_to(self) -> i8  { self as i8   } }
impl CastTo<i16> for f64 { fn cast_to(self) -> i16 { self as i16  } }
impl CastTo<i32> for f64 { fn cast_to(self) -> i32 { self as i32  } }
impl CastTo<i64> for f64 { fn cast_to(self) -> i64 { self as i64  } }
impl CastTo<f32> for f64 { fn cast_to(self) -> f32 { self as f32  } }
impl CastTo<f64> for f64 { fn cast_to(self) -> f64 { self as f64  } }
impl CastTo<c32> for f64 { fn cast_to(self) -> c32 { c32::new(self as f32, 0.0) } }
impl CastTo<c64> for f64 { fn cast_to(self) -> c64 { c64::new(self as f64, 0.0) } }

impl CastTo<u8>  for c32 { fn cast_to(self) -> u8  { self.re as u8   } }
impl CastTo<u16> for c32 { fn cast_to(self) -> u16 { self.re as u16  } }
impl CastTo<u32> for c32 { fn cast_to(self) -> u32 { self.re as u32  } }
impl CastTo<u64> for c32 { fn cast_to(self) -> u64 { self.re as u64  } }
impl CastTo<i8>  for c32 { fn cast_to(self) -> i8  { self.re as i8   } }
impl CastTo<i16> for c32 { fn cast_to(self) -> i16 { self.re as i16  } }
impl CastTo<i32> for c32 { fn cast_to(self) -> i32 { self.re as i32  } }
impl CastTo<i64> for c32 { fn cast_to(self) -> i64 { self.re as i64  } }
impl CastTo<f32> for c32 { fn cast_to(self) -> f32 { self.re as f32  } }
impl CastTo<f64> for c32 { fn cast_to(self) -> f64 { self.re as f64  } }
impl CastTo<c32> for c32 { fn cast_to(self) -> c32 { self } }
impl CastTo<c64> for c32 { fn cast_to(self) -> c64 { c64::new(self.re as f64, self.im as f64) } }

impl CastTo<u8>  for c64 { fn cast_to(self) -> u8  { self.re as u8   } }
impl CastTo<u16> for c64 { fn cast_to(self) -> u16 { self.re as u16  } }
impl CastTo<u32> for c64 { fn cast_to(self) -> u32 { self.re as u32  } }
impl CastTo<u64> for c64 { fn cast_to(self) -> u64 { self.re as u64  } }
impl CastTo<i8>  for c64 { fn cast_to(self) -> i8  { self.re as i8   } }
impl CastTo<i16> for c64 { fn cast_to(self) -> i16 { self.re as i16  } }
impl CastTo<i32> for c64 { fn cast_to(self) -> i32 { self.re as i32  } }
impl CastTo<i64> for c64 { fn cast_to(self) -> i64 { self.re as i64  } }
impl CastTo<f32> for c64 { fn cast_to(self) -> f32 { self.re as f32  } }
impl CastTo<f64> for c64 { fn cast_to(self) -> f64 { self.re as f64  } }
impl CastTo<c32> for c64 { fn cast_to(self) -> c32 { c32::new(self.re as f32, self.im as f32) } }
impl CastTo<c64> for c64 { fn cast_to(self) -> c64 { self } }

/// Trait allowing to cast T into Self.
pub trait CastFrom<T : Copy> {
    fn cast_from(t: T) -> Self;
}

impl CastFrom<u8>  for u8  { fn cast_from(t: u8)  -> u8  { t as u8  } }
impl CastFrom<u16> for u8  { fn cast_from(t: u16) -> u8  { t as u8  } }
impl CastFrom<u32> for u8  { fn cast_from(t: u32) -> u8  { t as u8  } }
impl CastFrom<u64> for u8  { fn cast_from(t: u64) -> u8  { t as u8  } }
impl CastFrom<i8>  for u8  { fn cast_from(t: i8)  -> u8  { t as u8  } }
impl CastFrom<i16> for u8  { fn cast_from(t: i16) -> u8  { t as u8  } }
impl CastFrom<i32> for u8  { fn cast_from(t: i32) -> u8  { t as u8  } }
impl CastFrom<i64> for u8  { fn cast_from(t: i64) -> u8  { t as u8  } }
impl CastFrom<f32> for u8  { fn cast_from(t: f32) -> u8  { t as u8  } }
impl CastFrom<f64> for u8  { fn cast_from(t: f64) -> u8  { t as u8  } }
impl CastFrom<c32> for u8  { fn cast_from(t: c32) -> u8  { t.re as u8  } }
impl CastFrom<c64> for u8  { fn cast_from(t: c64) -> u8  { t.re as u8  } }

impl CastFrom<u8>  for u16 { fn cast_from(t: u8)  -> u16 { t as u16 } }
impl CastFrom<u16> for u16 { fn cast_from(t: u16) -> u16 { t as u16 } }
impl CastFrom<u32> for u16 { fn cast_from(t: u32) -> u16 { t as u16 } }
impl CastFrom<u64> for u16 { fn cast_from(t: u64) -> u16 { t as u16 } }
impl CastFrom<i8>  for u16 { fn cast_from(t: i8)  -> u16 { t as u16 } }
impl CastFrom<i16> for u16 { fn cast_from(t: i16) -> u16 { t as u16 } }
impl CastFrom<i32> for u16 { fn cast_from(t: i32) -> u16 { t as u16 } }
impl CastFrom<i64> for u16 { fn cast_from(t: i64) -> u16 { t as u16 } }
impl CastFrom<f32> for u16 { fn cast_from(t: f32) -> u16 { t as u16 } }
impl CastFrom<f64> for u16 { fn cast_from(t: f64) -> u16 { t as u16 } }
impl CastFrom<c32> for u16 { fn cast_from(t: c32) -> u16 { t.re as u16 } }
impl CastFrom<c64> for u16 { fn cast_from(t: c64) -> u16 { t.re as u16 } }

impl CastFrom<u8>  for u32 { fn cast_from(t: u8)  -> u32 { t as u32 } }
impl CastFrom<u16> for u32 { fn cast_from(t: u16) -> u32 { t as u32 } }
impl CastFrom<u32> for u32 { fn cast_from(t: u32) -> u32 { t as u32 } }
impl CastFrom<u64> for u32 { fn cast_from(t: u64) -> u32 { t as u32 } }
impl CastFrom<i8>  for u32 { fn cast_from(t: i8)  -> u32 { t as u32 } }
impl CastFrom<i16> for u32 { fn cast_from(t: i16) -> u32 { t as u32 } }
impl CastFrom<i32> for u32 { fn cast_from(t: i32) -> u32 { t as u32 } }
impl CastFrom<i64> for u32 { fn cast_from(t: i64) -> u32 { t as u32 } }
impl CastFrom<f32> for u32 { fn cast_from(t: f32) -> u32 { t as u32 } }
impl CastFrom<f64> for u32 { fn cast_from(t: f64) -> u32 { t as u32 } }
impl CastFrom<c32> for u32 { fn cast_from(t: c32) -> u32 { t.re as u32 } }
impl CastFrom<c64> for u32 { fn cast_from(t: c64) -> u32 { t.re as u32 } }

impl CastFrom<u8>  for u64 { fn cast_from(t: u8)  -> u64 { t as u64 } }
impl CastFrom<u16> for u64 { fn cast_from(t: u16) -> u64 { t as u64 } }
impl CastFrom<u32> for u64 { fn cast_from(t: u32) -> u64 { t as u64 } }
impl CastFrom<u64> for u64 { fn cast_from(t: u64) -> u64 { t as u64 } }
impl CastFrom<i8>  for u64 { fn cast_from(t: i8)  -> u64 { t as u64 } }
impl CastFrom<i16> for u64 { fn cast_from(t: i16) -> u64 { t as u64 } }
impl CastFrom<i32> for u64 { fn cast_from(t: i32) -> u64 { t as u64 } }
impl CastFrom<i64> for u64 { fn cast_from(t: i64) -> u64 { t as u64 } }
impl CastFrom<f32> for u64 { fn cast_from(t: f32) -> u64 { t as u64 } }
impl CastFrom<f64> for u64 { fn cast_from(t: f64) -> u64 { t as u64 } }
impl CastFrom<c32> for u64 { fn cast_from(t: c32) -> u64 { t.re as u64 } }
impl CastFrom<c64> for u64 { fn cast_from(t: c64) -> u64 { t.re as u64 } }

impl CastFrom<u8>  for i8  { fn cast_from(t: u8)  -> i8  { t as i8  } }
impl CastFrom<u16> for i8  { fn cast_from(t: u16) -> i8  { t as i8  } }
impl CastFrom<u32> for i8  { fn cast_from(t: u32) -> i8  { t as i8  } }
impl CastFrom<u64> for i8  { fn cast_from(t: u64) -> i8  { t as i8  } }
impl CastFrom<i8>  for i8  { fn cast_from(t: i8)  -> i8  { t as i8  } }
impl CastFrom<i16> for i8  { fn cast_from(t: i16) -> i8  { t as i8  } }
impl CastFrom<i32> for i8  { fn cast_from(t: i32) -> i8  { t as i8  } }
impl CastFrom<i64> for i8  { fn cast_from(t: i64) -> i8  { t as i8  } }
impl CastFrom<f32> for i8  { fn cast_from(t: f32) -> i8  { t as i8  } }
impl CastFrom<f64> for i8  { fn cast_from(t: f64) -> i8  { t as i8  } }
impl CastFrom<c32> for i8  { fn cast_from(t: c32) -> i8  { t.re as i8  } }
impl CastFrom<c64> for i8  { fn cast_from(t: c64) -> i8  { t.re as i8  } }

impl CastFrom<u8>  for i16 { fn cast_from(t: u8)  -> i16 { t as i16 } }
impl CastFrom<u16> for i16 { fn cast_from(t: u16) -> i16 { t as i16 } }
impl CastFrom<u32> for i16 { fn cast_from(t: u32) -> i16 { t as i16 } }
impl CastFrom<u64> for i16 { fn cast_from(t: u64) -> i16 { t as i16 } }
impl CastFrom<i8>  for i16 { fn cast_from(t: i8)  -> i16 { t as i16 } }
impl CastFrom<i16> for i16 { fn cast_from(t: i16) -> i16 { t as i16 } }
impl CastFrom<i32> for i16 { fn cast_from(t: i32) -> i16 { t as i16 } }
impl CastFrom<i64> for i16 { fn cast_from(t: i64) -> i16 { t as i16 } }
impl CastFrom<f32> for i16 { fn cast_from(t: f32) -> i16 { t as i16 } }
impl CastFrom<f64> for i16 { fn cast_from(t: f64) -> i16 { t as i16 } }
impl CastFrom<c32> for i16 { fn cast_from(t: c32) -> i16 { t.re as i16 } }
impl CastFrom<c64> for i16 { fn cast_from(t: c64) -> i16 { t.re as i16 } }

impl CastFrom<u8>  for i32 { fn cast_from(t: u8)  -> i32 { t as i32 } }
impl CastFrom<u16> for i32 { fn cast_from(t: u16) -> i32 { t as i32 } }
impl CastFrom<u32> for i32 { fn cast_from(t: u32) -> i32 { t as i32 } }
impl CastFrom<u64> for i32 { fn cast_from(t: u64) -> i32 { t as i32 } }
impl CastFrom<i8>  for i32 { fn cast_from(t: i8)  -> i32 { t as i32 } }
impl CastFrom<i16> for i32 { fn cast_from(t: i16) -> i32 { t as i32 } }
impl CastFrom<i32> for i32 { fn cast_from(t: i32) -> i32 { t as i32 } }
impl CastFrom<i64> for i32 { fn cast_from(t: i64) -> i32 { t as i32 } }
impl CastFrom<f32> for i32 { fn cast_from(t: f32) -> i32 { t as i32 } }
impl CastFrom<f64> for i32 { fn cast_from(t: f64) -> i32 { t as i32 } }
impl CastFrom<c32> for i32 { fn cast_from(t: c32) -> i32 { t.re as i32 } }
impl CastFrom<c64> for i32 { fn cast_from(t: c64) -> i32 { t.re as i32 } }

impl CastFrom<u8>  for i64 { fn cast_from(t: u8)  -> i64 { t as i64 } }
impl CastFrom<u16> for i64 { fn cast_from(t: u16) -> i64 { t as i64 } }
impl CastFrom<u32> for i64 { fn cast_from(t: u32) -> i64 { t as i64 } }
impl CastFrom<u64> for i64 { fn cast_from(t: u64) -> i64 { t as i64 } }
impl CastFrom<i8>  for i64 { fn cast_from(t: i8)  -> i64 { t as i64 } }
impl CastFrom<i16> for i64 { fn cast_from(t: i16) -> i64 { t as i64 } }
impl CastFrom<i32> for i64 { fn cast_from(t: i32) -> i64 { t as i64 } }
impl CastFrom<i64> for i64 { fn cast_from(t: i64) -> i64 { t as i64 } }
impl CastFrom<f32> for i64 { fn cast_from(t: f32) -> i64 { t as i64 } }
impl CastFrom<f64> for i64 { fn cast_from(t: f64) -> i64 { t as i64 } }
impl CastFrom<c32> for i64 { fn cast_from(t: c32) -> i64 { t.re as i64 } }
impl CastFrom<c64> for i64 { fn cast_from(t: c64) -> i64 { t.re as i64 } }

impl CastFrom<u8>  for f32 { fn cast_from(t: u8)  -> f32 { t as f32 } }
impl CastFrom<u16> for f32 { fn cast_from(t: u16) -> f32 { t as f32 } }
impl CastFrom<u32> for f32 { fn cast_from(t: u32) -> f32 { t as f32 } }
impl CastFrom<u64> for f32 { fn cast_from(t: u64) -> f32 { t as f32 } }
impl CastFrom<i8>  for f32 { fn cast_from(t: i8)  -> f32 { t as f32 } }
impl CastFrom<i16> for f32 { fn cast_from(t: i16) -> f32 { t as f32 } }
impl CastFrom<i32> for f32 { fn cast_from(t: i32) -> f32 { t as f32 } }
impl CastFrom<i64> for f32 { fn cast_from(t: i64) -> f32 { t as f32 } }
impl CastFrom<f32> for f32 { fn cast_from(t: f32) -> f32 { t as f32 } }
impl CastFrom<f64> for f32 { fn cast_from(t: f64) -> f32 { t as f32 } }
impl CastFrom<c32> for f32 { fn cast_from(t: c32) -> f32 { t.re as f32 } }
impl CastFrom<c64> for f32 { fn cast_from(t: c64) -> f32 { t.re as f32 } }

impl CastFrom<u8>  for f64 { fn cast_from(t: u8)  -> f64 { t as f64 } }
impl CastFrom<u16> for f64 { fn cast_from(t: u16) -> f64 { t as f64 } }
impl CastFrom<u32> for f64 { fn cast_from(t: u32) -> f64 { t as f64 } }
impl CastFrom<u64> for f64 { fn cast_from(t: u64) -> f64 { t as f64 } }
impl CastFrom<i8>  for f64 { fn cast_from(t: i8)  -> f64 { t as f64 } }
impl CastFrom<i16> for f64 { fn cast_from(t: i16) -> f64 { t as f64 } }
impl CastFrom<i32> for f64 { fn cast_from(t: i32) -> f64 { t as f64 } }
impl CastFrom<i64> for f64 { fn cast_from(t: i64) -> f64 { t as f64 } }
impl CastFrom<f32> for f64 { fn cast_from(t: f32) -> f64 { t as f64 } }
impl CastFrom<f64> for f64 { fn cast_from(t: f64) -> f64 { t as f64 } }
impl CastFrom<c32> for f64 { fn cast_from(t: c32) -> f64 { t.re as f64 } }
impl CastFrom<c64> for f64 { fn cast_from(t: c64) -> f64 { t.re as f64 } }

impl CastFrom<u8>  for c32 { fn cast_from(t: u8)  -> c32 { c32::new(t as f32, 0.0) } }
impl CastFrom<u16> for c32 { fn cast_from(t: u16) -> c32 { c32::new(t as f32, 0.0) } }
impl CastFrom<u32> for c32 { fn cast_from(t: u32) -> c32 { c32::new(t as f32, 0.0) } }
impl CastFrom<u64> for c32 { fn cast_from(t: u64) -> c32 { c32::new(t as f32, 0.0) } }
impl CastFrom<i8>  for c32 { fn cast_from(t: i8)  -> c32 { c32::new(t as f32, 0.0) } }
impl CastFrom<i16> for c32 { fn cast_from(t: i16) -> c32 { c32::new(t as f32, 0.0) } }
impl CastFrom<i32> for c32 { fn cast_from(t: i32) -> c32 { c32::new(t as f32, 0.0) } }
impl CastFrom<i64> for c32 { fn cast_from(t: i64) -> c32 { c32::new(t as f32, 0.0) } }
impl CastFrom<f32> for c32 { fn cast_from(t: f32) -> c32 { c32::new(t as f32, 0.0) } }
impl CastFrom<f64> for c32 { fn cast_from(t: f64) -> c32 { c32::new(t as f32, 0.0) } }
impl CastFrom<c32> for c32 { fn cast_from(t: c32) -> c32 { t } }
impl CastFrom<c64> for c32 { fn cast_from(t: c64) -> c32 { c32::new(t.re as f32, t.im as f32) } }

impl CastFrom<u8>  for c64 { fn cast_from(t: u8)  -> c64 { c64::new(t as f64, 0.0) } }
impl CastFrom<u16> for c64 { fn cast_from(t: u16) -> c64 { c64::new(t as f64, 0.0) } }
impl CastFrom<u32> for c64 { fn cast_from(t: u32) -> c64 { c64::new(t as f64, 0.0) } }
impl CastFrom<u64> for c64 { fn cast_from(t: u64) -> c64 { c64::new(t as f64, 0.0) } }
impl CastFrom<i8>  for c64 { fn cast_from(t: i8)  -> c64 { c64::new(t as f64, 0.0) } }
impl CastFrom<i16> for c64 { fn cast_from(t: i16) -> c64 { c64::new(t as f64, 0.0) } }
impl CastFrom<i32> for c64 { fn cast_from(t: i32) -> c64 { c64::new(t as f64, 0.0) } }
impl CastFrom<i64> for c64 { fn cast_from(t: i64) -> c64 { c64::new(t as f64, 0.0) } }
impl CastFrom<f32> for c64 { fn cast_from(t: f32) -> c64 { c64::new(t as f64, 0.0) } }
impl CastFrom<f64> for c64 { fn cast_from(t: f64) -> c64 { c64::new(t as f64, 0.0) } }
impl CastFrom<c32> for c64 { fn cast_from(t: c32) -> c64 { c64::new(t.re as f64, t.im as f64) } }
impl CastFrom<c64> for c64 { fn cast_from(t: c64) -> c64 { t } }
