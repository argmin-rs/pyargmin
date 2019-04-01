// Copyright 2019 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

mod executor;
mod operator;

use crate::operator::*;
use argmin::prelude::*;
use argmin::solver::landweber::Landweber;
use executor::*;
use ndarray::{array, Array1};
use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::class::methods::{PyMethodDefType, PyMethodsProtocol};
use pyo3::prelude::*;
use pyo3::type_object::PyTypeInfo;
use pyo3::wrap_pyfunction;

#[pyfunction]
/// blah
fn closure(obj: PyObject) -> PyResult<()> {
    let func = PyArgminOp::new(&obj);
    let out = func.apply(&array![1.0f64, 2.0f64]);
    println!("Rust: {:?}", out);
    Ok(())
}

#[pyclass(name=Landweber)]
struct PyLandweber {
    solver: Landweber,
}

#[pymethods]
impl PyLandweber {
    #[new]
    fn new(obj: &PyRawObject, omega: f64) {
        obj.init({
            PyLandweber {
                solver: Landweber::new(omega),
            }
        });
    }

    fn set_omega(&mut self, omega: f64) {
        println!("fufufu_{}", omega)
    }
}

#[pyfunction]
/// blah
fn landweber(omega: f64) -> Py<PyLandweber> {
    let gil_guard = Python::acquire_gil();
    let py = gil_guard.python();
    Py::new(
        py,
        PyLandweber {
            solver: Landweber::new(omega),
        },
    )
    .unwrap()
}

#[pyfunction]
/// blah
fn closure3(func: PyObject) -> PyResult<()> {
    let gil_guard = Python::acquire_gil();
    let py = gil_guard.python();
    let blah = Array1::from_vec(vec![1.0f64, 2.0, 3.0]);
    func.call1(py, (blah.into_pyarray(py),))?;
    Ok(())
}

// #[pyfunction]
// /// Get an executor
// // pub fn new(op: O, solver: S, init_param: O::Param) -> Self {
// fn executor(
//     op: PyArgminOp,
//     solver: PyObject,
//     init_param: PyObject,
// ) -> PyResult<Executor<PyArgminOp, PyObject>> {
//     Ok(Executor::new(op, solver, init_param))
// }

/// python module
#[pymodule]
fn pyargmin(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(closure))?;
    m.add_wrapped(wrap_pyfunction!(closure3))?;
    m.add_wrapped(wrap_pyfunction!(landweber))?;
    Ok(())
}
