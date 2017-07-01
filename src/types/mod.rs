use std::fmt::Display;

/// Module implementing floating point complex type compatible with BLAS.
pub mod complex;
/// Module implementing Cast trait for numerical types.
pub mod cast;

pub use types::complex::{Complex, c32, c64};
pub use types::cast::{CastTo, CastFrom};

/// Enumeration for the numerical type supported by RDS.
pub enum RDSType {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    C32,
    C64
}

/// Trait implemented by all the RDS supported types.
pub trait RDSTyped : CastTo<u8> + CastTo<u16> + CastTo<u32> + CastTo<u64> + 
                     CastTo<i8> + CastTo<i16> + CastTo<i32> + CastTo<i64> + 
                     CastTo<f32> + CastTo<f64> + CastTo<c32> + CastTo<c64> + 
                     CastFrom<u8> + CastFrom<u16> + CastFrom<u32> + CastFrom<u64> + 
                     CastFrom<i8> + CastFrom<i16> + CastFrom<i32> + CastFrom<i64> + 
                     CastFrom<f32> + CastFrom<f64> + CastFrom<c32> + CastFrom<c64> + 
                     Clone + Copy + Display {
    /// Reflection function which allow to query the type in a generic context.
    fn rds_type() -> RDSType;
}

impl RDSTyped for u8 {
    fn rds_type() -> RDSType {
        RDSType::U8
    }
}

impl RDSTyped for u16 {
    fn rds_type() -> RDSType {
        RDSType::U16
    }
}

impl RDSTyped for u32 {
    fn rds_type() -> RDSType {
        RDSType::U32
    }
}

impl RDSTyped for u64 {
    fn rds_type() -> RDSType {
        RDSType::U64
    }
}

impl RDSTyped for i8 {
    fn rds_type() -> RDSType {
        RDSType::I8
    }
}

impl RDSTyped for i16 {
    fn rds_type() -> RDSType {
        RDSType::I16
    }
}

impl RDSTyped for i32 {
    fn rds_type() -> RDSType {
        RDSType::I32
    }
}

impl RDSTyped for i64 {
    fn rds_type() -> RDSType {
        RDSType::I64
    }
}

impl RDSTyped for f32 {
    fn rds_type() -> RDSType {
        RDSType::F32
    }
}

impl RDSTyped for f64 {
    fn rds_type() -> RDSType {
        RDSType::F64
    }
}

impl RDSTyped for c32 {
    fn rds_type() -> RDSType {
        RDSType::C32
    }
}

impl RDSTyped for c64 {
    fn rds_type() -> RDSType {
        RDSType::C64
    }
}
