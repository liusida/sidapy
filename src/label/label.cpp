#include <iostream>
#include <stdio.h>

#include "label.h"
#include "mit_rahimi.h"

py::array_t<u_int64_t> label(py::array_t<int8_t> map) {
        u_int64_t *out_uc = new u_int64_t[map.shape(0) * map.shape(1)];
        ConnectedComponents cc(30);
        cc.connected(map.data(), out_uc, map.shape(0), map.shape(1),
                     std::equal_to<int8_t>(), false, (int8_t)1);

        // Create a Python object that will free the allocated
        // memory when destroyed:
        py::capsule free_when_done(out_uc, [](void *f) {
                double *foo = reinterpret_cast<double *>(f);
                // debug output
                // std::cerr << "Element [0] = " << foo[0] << "\n";
                // std::cerr << "freeing memory @ " << f << "\n";
                delete[] foo;
        });

        return py::array_t<u_int64_t>(
            {map.shape(0), map.shape(1)}, // shape
            {map.shape(0) * sizeof(u_int64_t),
             sizeof(u_int64_t)}, // C-style contiguous strides for double
            out_uc,              // the data pointer
            free_when_done);     // numpy array references this parent
}