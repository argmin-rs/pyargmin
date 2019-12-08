// Copyright 2019-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::operator::ParamKind;
use crate::operator::PyArgminOp;
use crate::PyLandweber;
use argmin::prelude::*;
use argmin::solver::landweber::Landweber;
use numpy::*;
use pyo3::prelude::*;

#[pyclass(name=Executor)]
pub struct PyExecutor {
    pub exec: Executor<PyArgminOp, Landweber>,
}

#[pymethods]
impl PyExecutor {
    // pub fn new(op: O, solver: S, init_param: O::Param) -> Self {
    #[new]
    fn new(
        obj: &PyRawObject,
        op: PyObject,
        solver: &mut PyLandweber,
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

    fn run(&self) -> PyResult<()> {
        let bla = self.exec.clone();
        let res = bla.run().unwrap();
        println!("{}", res);
        Ok(())
    }

    // fn max_iters(mut self, iters: u64) {
    //     self.exec = self.exec.max_iters(iters);
    // }
}
