// Copyright 2019 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::prelude::*;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct PyArgminOp {
    #[serde(skip)]
    obj: Option<PyObject>,
}

impl PyArgminOp {
    /// Constructor
    pub fn new(obj: PyObject) -> Self {
        PyArgminOp { obj: Some(obj) }
    }
}

impl Clone for PyArgminOp {
    fn clone(&self) -> Self {
        let gil_guard = Python::acquire_gil();
        let py = gil_guard.python();
        PyArgminOp {
            obj: Some(self.obj.as_ref().unwrap().clone_ref(py)),
        }
    }
}

unsafe impl Send for PyArgminOp {}
unsafe impl Sync for PyArgminOp {}

impl ArgminOp for PyArgminOp
where
    PyArgminOp: Clone,
{
    type Param = Vec<f64>;
    type Output = f64;
    type Hessian = ();

    fn apply(&self, x: &Self::Param) -> Result<Self::Output, Error> {
        let gil_guard = Python::acquire_gil();
        let py = gil_guard.python();
        let out: f64 = self
            .obj
            .as_ref()
            .unwrap()
            .call_method1(py, "apply", ((x[0], x[1]),))
            .map_err(|e| ArgminError::ImpossibleError {
                text: format!("{:?}", e).to_string(),
            })?
            .extract(py)
            .map_err(|e| ArgminError::ImpossibleError {
                text: format!("{:?}", e).to_string(),
            })?;
        // println!("{:?}", out);
        Ok(out)
    }
}
