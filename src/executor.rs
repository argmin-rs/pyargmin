// Copyright 2019-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
use std::f64;

use crate::operator::ParamKind;
use crate::operator::PyArgminOp;
use crate::{LBFGS_type, PyLBFGS};
use argmin::prelude::*;
use argmin::solver::quasinewton::LBFGS;
use ndarray::arr1;
use numpy::*;
use pyo3::prelude::*;

#[pyclass(name=Executor)]
pub struct PyExecutor {
    pub exec: Executor<PyArgminOp, LBFGS_type>,
}

#[pymethods]
impl PyExecutor {
    // pub fn new(op: O, solver: S, init_param: O::Param) -> Self {
    #[new]
    fn new(
        obj: &PyRawObject,
        op: PyObject,
        solver: &mut PyLBFGS,
        init_param: PyObject,
    ) -> PyResult<()> {
        let gil_guard = Python::acquire_gil();
        let py = gil_guard.python();
        let init_param: &PyArray<f64, Ix1> = init_param.extract(py)?;
        obj.init({
            PyExecutor {
                exec: Executor::new(
                    op.extract(py)?,
                    solver.inner(),
                    ParamKind::Ndarray(init_param.as_array_mut().to_owned()),
                )
                .max_iters(20), // .add_observer(ArgminSlogLogger::term(), ObserverMode::Always),
            }
        });
        Ok(())
    }

    fn run(&self) -> PyResult<(Py<PyArray1<f64>>)> {
        let executor = self.exec.clone();
        let res = executor.run().unwrap();
        let x = match res.state.param {
            ParamKind::Ndarray(value) => value,
            _ => arr1(&[f64::NAN, f64::NAN]),
        };
        let gil = Python::acquire_gil();
        let x = PyArray1::from_array(gil.python(), &x);
        Ok(x.to_owned())
    }

    // fn max_iters(mut self, iters: u64) {
    //     self.exec = self.exec.max_iters(iters);
    // }
}
