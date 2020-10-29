#include <pybind11/pybind11.h>

#include "hw7q3_forest_fire/hw7q3_forest_fire.h"
#include "add/add.h"
#include "label/label.h"
#include "my_func/my_func.h"

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

    py::class_<hw7q3_forest_fire>(m, "hw7q3_forest_fire").def(py::init<const int>()).def("set_spark_prob", &hw7q3_forest_fire::set_spark_prob).def("avg_forest_fire_size", &hw7q3_forest_fire::avg_forest_fire_size);

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
