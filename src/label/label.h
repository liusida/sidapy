#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
namespace py = pybind11;

py::array_t<u_int64_t> label(py::array_t<int8_t> map);
