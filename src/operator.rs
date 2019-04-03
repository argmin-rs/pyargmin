// Copyright 2019 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::prelude::*;
// use ndarray::Array1;
use numpy::*;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use serde::{Deserialize, Serialize};
// use std::any::Any;

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

#[derive(Clone, Serialize, Deserialize)]
pub enum ParamType {
    Ndarray(ndarray::Array1<f64>),
    Other,
}

impl std::default::Default for ParamType {
    fn default() -> Self {
        ParamType::Other
    }
}

impl ArgminMul<ParamType, ParamType> for f64 {
    #[inline]
    fn mul(&self, other: &ParamType) -> ParamType {
        if let ParamType::Ndarray(ref y) = other {
            ParamType::Ndarray(self.mul(y))
        } else {
            unimplemented!()
        }
    }
}

impl ArgminSub<ParamType, ParamType> for ParamType {
    #[inline]
    fn sub(&self, other: &ParamType) -> ParamType {
        if let (ParamType::Ndarray(ref x), ParamType::Ndarray(ref y)) = (self, other) {
            ParamType::Ndarray(x.sub(y))
        } else {
            unimplemented!()
        }
    }
}

impl ParamType {
    pub fn ndarray(&self) -> Option<ndarray::Array1<f64>> {
        if let ParamType::Ndarray(ref x) = self {
            Some(x.clone())
        } else {
            None
        }
    }

    pub fn other(&self) -> Option<()> {
        if let ParamType::Other = *self {
            Some(())
        } else {
            None
        }
    }
}

impl ArgminOp for PyArgminOp
where
    PyArgminOp: Clone,
{
    type Param = ParamType;
    type Output = f64;
    type Hessian = ();

    fn apply(&self, x: &Self::Param) -> Result<Self::Output, Error> {
        let gil_guard = Python::acquire_gil();
        let py = gil_guard.python();
        let param = match x {
            ParamType::Ndarray(ref x) => x.to_pyarray(py),
            _ => unimplemented!(),
        };

        let out: f64 = self
            .obj
            .as_ref()
            .unwrap()
            .call_method1(py, "apply", (param,))
            .map_err(|e| ArgminError::NotImplemented {
                text: format!("apply method is not implemented: {:?}", e).to_string(),
            })?
            .extract(py)
            .map_err(|e| ArgminError::ImpossibleError {
                text: format!("Wrong return type from apply method: {:?}", e).to_string(),
            })?;
        Ok(out)
    }

    fn gradient(&self, x: &Self::Param) -> Result<Self::Param, Error> {
        let gil_guard = Python::acquire_gil();
        let py = gil_guard.python();
        let param = match x {
            ParamType::Ndarray(ref x) => x.to_pyarray(py),
            _ => unimplemented!(),
        };

        let bla = self
            .obj
            .as_ref()
            .unwrap()
            .call_method1(py, "gradient", (param,))
            .map_err(|e| ArgminError::NotImplemented {
                text: format!("gradient method is not implemented: {:?}", e).to_string(),
            })?;
        let out: &PyArray1<f64> = bla.extract(py).map_err(|e| ArgminError::ImpossibleError {
            text: format!("Wrong return type from apply method: {:?}", e).to_string(),
        })?;
        Ok(ParamType::Ndarray(out.as_array_mut().to_owned()))
    }
}
