#include <pybind11/pybind11.h>

#include "label/label.h"
#include "my_func/my_func.h"
#include "add/add.h"

// namespace py = pybind11;

PYBIND11_MODULE(sidapy, m) {
    m.doc() = R"pbdoc(
        SidaPy: my personal bindings to C++ and CUDA
        -----------------------

        .. currentmodule:: sidapy

        .. autosummary::
           :toctree: _generate

           add
           my_func
    )pbdoc";
    
    m.def("label", &label, R"pbdoc(
        Similar to scipy.ndimage.label.
    )pbdoc");

    m.def("my_func", &my_func, R"pbdoc(
        My test function from cuda file.
    )pbdoc");

    m.def("add", &add, R"pbdoc(
        Add two numbers.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
