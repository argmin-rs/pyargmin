// use ndarray::Array1;
// use numpy::{IntoPyArray, PyArrayDyn};
use argmin::prelude::*;
use failure::Error;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct PyArgminOp {
    #[serde(skip)]
    obj: Option<PyObject>,
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
            .unwrap()
            .extract(py)
            .unwrap();
        println!("{:?}", out);
        Ok(out)
    }
}

#[pyfunction]
/// blah
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
/// blah
fn closure(func: PyObject) -> PyResult<()> {
    let gil_guard = Python::acquire_gil();
    let py = gil_guard.python();
    func.call0(py)?;
    Ok(())
}

#[pyfunction]
/// blah
fn closure2(func: PyObject) -> PyResult<()> {
    let gil_guard = Python::acquire_gil();
    let py = gil_guard.python();
    func.call1(py, (2.56f64,))?;
    Ok(())
}

#[pyfunction]
/// blah
fn closure3(func: PyObject) -> PyResult<()> {
    let gil_guard = Python::acquire_gil();
    let py = gil_guard.python();
    let func = PyArgminOp { obj: Some(func) };
    let out = func.apply(&vec![1.0f64, 2.0f64]);
    // let out = <PyObject as ArgminOp>::apply(&func, vec![1.0f64, 2.0f64]);
    // let out: f64 = func
    //     .call_method1(py, "apply", ((1.0f64, 2.0f64),))?
    //     .extract(py)?;
    println!("{:?}", out);
    Ok(())
}

// #[pyfunction]
// /// blah
// fn closure3(func: PyObject) -> PyResult<()> {
//     let gil_guard = Python::acquire_gil();
//     let py = gil_guard.python();
//     let blah = Array1::from_vec(vec![1.0f64, 2.0, 3.0]);
//     func.call1(py, (blah.into_pyarray(py),))?;
//     Ok(())
// }

/// python module
#[pymodule]
fn pyargmin(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(sum_as_string))?;
    m.add_wrapped(wrap_pyfunction!(closure))?;
    m.add_wrapped(wrap_pyfunction!(closure2))?;
    m.add_wrapped(wrap_pyfunction!(closure3))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
