// Copyright 2019-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

mod executor;
mod operator;

use crate::operator::ParamKind;
use crate::operator::*;
use argmin::prelude::*;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use executor::*;
use ndarray::{array, Array1};
use numpy::*;
use pyo3::prelude::*;
// use pyo3::type_object::PyTypeInfo;
use pyo3::wrap_pyfunction;

type LBFGS_type = LBFGS<MoreThuenteLineSearch<ParamKind>, ParamKind>;

#[pyfunction]
/// blah
fn closure(obj: PyObject) -> PyResult<()> {
    let func = PyArgminOp::new(&obj);
    let out = func.apply(&ParamKind::Ndarray(array![1.0f64, 2.0f64]));
    println!("Rust: {:?}", out);
    Ok(())
}

#[pyclass(name=LBFGS)]
struct PyLBFGS {
    solver: LBFGS_type,
}

#[pymethods]
impl PyLBFGS {
    #[new]
    fn new(obj: &PyRawObject, m: usize) {
        obj.init({
            PyLBFGS {
                solver: LBFGS::new(MoreThuenteLineSearch::new(), m),
            }
        });
    }
}

impl PyLBFGS {
    fn inner(&self) -> LBFGS_type {
        self.solver.clone()
    }
}

#[pyfunction]
/// blah
fn lbfgs(m: usize) -> Py<PyLBFGS> {
    let gil_guard = Python::acquire_gil();
    let py = gil_guard.python();
    Py::new(
        py,
        PyLBFGS {
            solver: LBFGS::new(MoreThuenteLineSearch::new(), m),
        },
    )
    .unwrap()
}

#[pyfunction]
/// blah
fn closure3(func: PyObject) -> PyResult<()> {
    let gil_guard = Python::acquire_gil();
    let py = gil_guard.python();
    let blah = Array1::from(vec![1.0f64, 2.0, 3.0]);
    func.call1(py, (blah.into_pyarray(py),))?;
    Ok(())
}

#[pyfunction(max_iter = 20)]
/// executor(op, solver, init_param, max_iter=20)
///
/// Get an executor
fn executor(op: PyObject, solver: &mut PyLBFGS, init_param: PyObject, max_iter: u64) -> Py<PyExecutor> {
    let gil_guard = Python::acquire_gil();
    let py = gil_guard.python();
    let init_param: &PyArray<f64, Ix1> = init_param.extract(py).unwrap();
    Py::new(
        py,
        PyExecutor {
            exec: Executor::new(
                op.extract(py).unwrap(),
                solver.inner(),
                ParamKind::Ndarray(init_param.as_array_mut().to_owned()),
            )
            .max_iters(max_iter)
            .add_observer(ArgminSlogLogger::term(), ObserverMode::Always),
        },
    )
    .unwrap()
}

/// python module
#[pymodule]
fn _lib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(closure))?;
    m.add_wrapped(wrap_pyfunction!(closure3))?;
    m.add_wrapped(wrap_pyfunction!(lbfgs))?;
    m.add_wrapped(wrap_pyfunction!(executor))?;
    Ok(())
}
