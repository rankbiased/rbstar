use pyo3::prelude::*;

#[pyfunction]
fn rb_function() {}

#[pyclass]
struct RBStruct {}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _rbpy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rb_function, m)?)?;
    m.add_class::<RBStruct>()?;
    Ok(())
}
