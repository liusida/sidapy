#if !defined(HW7Q3_FOREST_FIRE_H)
#define HW7Q3_FOREST_FIRE_H

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
namespace py = pybind11;

class hw7q3_forest_fire {
  private:
    /* data */
    bool _verbose;
    float *_ptr_spark_prob;
    ssize_t _map_X, _map_Y;
    int32_t *d_great_map;
    ssize_t *ret_fire_size;
    ssize_t *_event_size; // to save ret_fire_size at certain step

  public:
    hw7q3_forest_fire(const int verbose);
    ~hw7q3_forest_fire();

    void set_spark_prob(py::array_t<float> prob);
    float avg_forest_fire_size(py::array_t<int32_t> map);
};

#endif // HW7Q3_FOREST_FIRE_H
