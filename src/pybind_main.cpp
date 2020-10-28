#include <pybind11/pybind11.h>

#include <my_func/my_func.h>
#include <add/add.h>

namespace py = pybind11;

PYBIND11_MODULE(sidapy, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: sidapy

        .. autosummary::
           :toctree: _generate

           add
           my_func
    )pbdoc";
    
    m.def("my_func", &my_func, R"pbdoc(
        my function from cuda file.
    )pbdoc");

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
