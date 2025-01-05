pub const MR: usize = 4;
pub const NR: usize = 4;

// pub const NUM_THREADS: usize = 8;
// pub const MC: usize = MR * NUM_THREADS;
// pub const NC: usize = NR * NUM_THREADS;
// pub const KC: usize = NUM_THREADS;

pub const MC: usize = 16;
pub const NC: usize = 102;
pub const KC: usize = 500;

pub mod kernels;
pub mod mul;
pub mod mul_dp;
