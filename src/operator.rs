// Copyright 2019 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::prelude::*;
use ndarray::Array1;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct PyArgminOp {
    #[serde(skip)]
    obj: Option<PyObject>,
}

impl PyArgminOp {
    /// Constructor
    pub fn new(obj: &PyObject) -> Self {
        let gil_guard = Python::acquire_gil();
        let py = gil_guard.python();
        PyArgminOp {
            obj: Some(obj.clone_ref(py)),
        }
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

impl<'source> FromPyObject<'source> for PyArgminOp {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let gil_guard = Python::acquire_gil();
        let py = gil_guard.python();
        Ok(PyArgminOp::new(&ob.to_object(py)))
    }
}

unsafe impl Send for PyArgminOp {}
unsafe impl Sync for PyArgminOp {}

impl ArgminOp for PyArgminOp
where
    PyArgminOp: Clone,
{
    type Param = Array1<f64>;
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
            .map_err(|e| ArgminError::NotImplemented {
                text: format!("apply method is not implemented: {:?}", e).to_string(),
            })?
            .extract(py)
            .map_err(|e| ArgminError::ImpossibleError {
                text: format!("Wrong return type from apply method: {:?}", e).to_string(),
            })?;
        Ok(out)
    }
}
