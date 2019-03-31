// Copyright 2019 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// use ndarray::Array1;
// use numpy::{IntoPyArray, PyArrayDyn};
mod operator;

use crate::operator::*;
use argmin::prelude::*;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
/// blah
fn closure(func: PyObject) -> PyResult<()> {
    let func = PyArgminOp::new(func);
    let out = func.apply(&vec![1.0f64, 2.0f64]);
    println!("Rust: {:?}", out);
    Ok(())
}

/// python module
#[pymodule]
fn pyargmin(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(closure))?;
    Ok(())
}
