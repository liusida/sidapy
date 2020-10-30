#include <math.h> /* sqrt */

#include "common.cuh"

#include "hw7q3_forest_fire.h"

template <class INT> __host__ __device__ void _print_map(const INT *map, ssize_t _map_X, ssize_t _map_Y) {
    // for debug use
    for (ssize_t i = 0; i < _map_X; i++) {
        for (ssize_t j = 0; j < _map_Y; j++) {
            if (sizeof(INT) > sizeof(int)) {
                printf("%ld ", (long int)map[i * _map_X + j]);
            } else {
                printf("%d ", (int)map[i * _map_X + j]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

__device__ ssize_t _dfs(int32_t *d_map_to_burn, ssize_t x, ssize_t y, ssize_t _map_X, ssize_t _map_Y) {
    // DFS for calculating one forest fire size
    if (x < 0 || y < 0 || x >= _map_X || y >= _map_Y) { // out of edge
        return 0;
    }

    if (d_map_to_burn[x * _map_X + y] <= 0) { // no tree or burned
        d_map_to_burn[x * _map_X + y] = -1;   // burn the current grid
        return 0;
    }

    // printf("<%ld, %ld>", x, y);

    d_map_to_burn[x * _map_X + y] = -1; // burn the current grid

    return 1 + _dfs(d_map_to_burn, x - 1, y, _map_X, _map_Y) + _dfs(d_map_to_burn, x, y - 1, _map_X, _map_Y) + _dfs(d_map_to_burn, x + 1, y, _map_X, _map_Y) +
           _dfs(d_map_to_burn, x, y + 1, _map_X, _map_Y);
}

__device__ ssize_t count_same_continuous_label(int32_t *map, ssize_t x, ssize_t y, ssize_t _map_X, ssize_t _map_Y) {
    // spiral out from (x,y), more efficient
    // Unforturenately, this method is no faster than the simple counting.
    int current_direction = 0; // [0-4] up, right, down, left
    ssize_t current_x, current_y;
    current_x = x;
    current_y = y;
    ssize_t current_limb_length = 0;
    ssize_t current_spiral_diameter = 1;

    int32_t spark = map[x * _map_X + y];
    if (spark == 0)
        return 0;
    ssize_t ret = 0;
    ssize_t total_grid = _map_X * _map_Y;
    ssize_t i = 0;
    bool keep_searching = true;
    bool hit[4];
    hit[0] = hit[1] = hit[2] = hit[3] = false;
    while (true && keep_searching) {
        if (map[current_x * _map_X + current_y] == spark) {
            ret++;
            if (!hit[current_direction])
                hit[current_direction] = true;
        }
        if (++i >= total_grid) // all grid checked, no need to search any more. if you keep going, it will always end in invalid grid and loop for every.
            break;
        while (true && keep_searching) {
            if (current_direction == 0) {
                current_y--;
            } else if (current_direction == 1) {
                current_x++;
            } else if (current_direction == 2) {
                current_y++;
            } else if (current_direction == 3) {
                current_x--;
            }
            current_limb_length++;
            if (current_limb_length >= current_spiral_diameter) {
                if (hit[0] == false && hit[1] == false && hit[2] == false &&
                    hit[3] == false) { // all four sides checked, early stop. This early stop is why we want this spiral style checking.
                    keep_searching = false;
                    break;
                }
                current_direction = (current_direction + 1) % 4;
                hit[current_direction] = false;
                current_limb_length = 0;
                if (current_direction % 2 == 0) {
                    current_spiral_diameter++;
                }
            }
            if (current_x >= 0 && current_y >= 0 && current_x < _map_X && current_y < _map_Y) { // in valid grid
                break;
            } // otherwise move to the next grid
        }
    }
    return ret;
}

__device__ ssize_t count_same_label(int32_t *map, ssize_t x, ssize_t y, ssize_t _map_X, ssize_t _map_Y) {
    int32_t spark = map[x * _map_X + y];
    if (spark == 0)
        return 0;
    ssize_t ret = 0;
    for (ssize_t i = 0; i < _map_X * _map_Y; i++) {
        if (map[i] == spark && map[i] != 0)
            ret++;
    }
    return ret;
}

__global__ void fire_size(int32_t *d_great_map, ssize_t _map_X, ssize_t _map_Y, ssize_t *ret_fire_size, bool verbose) {
    ssize_t i = blockIdx.x * blockDim.x + threadIdx.x;
    ssize_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < _map_X && j < _map_Y) {
        // printf("(%ld, %ld)- ", i, j);

        int32_t *d_map_to_burn = &d_great_map[(i * _map_X + j) * _map_X * _map_Y];

        if (i == 1 && j == 0) {
            if (verbose) {
                printf("d_map_to_burn\n");
                // printf("<%ld, %ld>, ", i, j);
                _print_map(d_map_to_burn, _map_X, _map_Y);
            }
        }
        ret_fire_size[i * _map_X + j] = count_same_label(d_map_to_burn, i, j, _map_X, _map_Y);
        // printf("(%ld, %ld), ", i, j);
        if (i == 1 && j == 0) {
            if (verbose) {
                printf("d_map_to_burn after burnt\n");
                // printf("<%ld, %ld>, ", i, j);
                _print_map(d_map_to_burn, _map_X, _map_Y);
            }
        }
    }
}

__global__ void populate_great_map(int32_t *d_great_map, ssize_t _map_X, ssize_t _map_Y) {
    ssize_t i = blockIdx.x * blockDim.x + threadIdx.x;
    ssize_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < _map_X && j < _map_Y) {
        if (i == 0 && j == 0) { // the first frame has already been populated.
            return;
        }
        memcpy(&d_great_map[(i * _map_X + j) * _map_X * _map_Y], &d_great_map[0], _map_X * _map_Y * sizeof(int32_t));
    }
}

hw7q3_forest_fire::hw7q3_forest_fire(const int verbose) {
    _verbose = (bool)verbose;
    _ptr_spark_prob = nullptr;
    _map_X = 0;
    _map_Y = 0;
    d_great_map = nullptr;
    ret_fire_size = nullptr;
    // only need to enlarge stack memory if recursive _dfs() is used.
    // size_t pValue;
    // cudaDeviceGetLimit(&pValue, cudaLimitStackSize);
    // printf("cudaLimitStackSize %ld.\n", pValue);
    // gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize, pow(2, 17)));
    // cudaDeviceGetLimit(&pValue, cudaLimitStackSize);
    // printf("set cudaLimitStackSize to %ld.\n", pValue); // Because recursive function used. should avoid that.
}

hw7q3_forest_fire::~hw7q3_forest_fire() {
    if (_ptr_spark_prob != nullptr) {
        delete[] _ptr_spark_prob;
        _ptr_spark_prob = nullptr;
    }
    if (d_great_map != nullptr) {
        gpuErrchk(cudaFree(d_great_map));
        d_great_map = nullptr;
    }
    if (ret_fire_size != nullptr) {
        gpuErrchk(cudaFree(ret_fire_size));
        ret_fire_size = nullptr;
    }
    if (_verbose)
        printf("cleaned.\n");
}

void hw7q3_forest_fire::set_spark_prob(py::array_t<float> prob) {
    _map_X = prob.shape(0);
    _map_Y = prob.shape(1);
    if (_ptr_spark_prob != nullptr) {
        delete[] _ptr_spark_prob;
        _ptr_spark_prob = nullptr;
    }
    _ptr_spark_prob = new float[_map_X * _map_Y];
    memcpy(_ptr_spark_prob, prob.data(), _map_X * _map_Y * sizeof(float));
}

float hw7q3_forest_fire::avg_forest_fire_size(py::array_t<int32_t> map) {
    if (_ptr_spark_prob != nullptr) {
        if (_verbose)
            printf("Starting avg_forest_fire_size...\n");

        int blockSize;
        int minGridSize;

        if (d_great_map == nullptr) {
            // make a great map, which has x*y maps in it, and burn them parallelly.
            gpuErrchk(cudaMallocManaged(&d_great_map, _map_X * _map_Y * _map_X * _map_Y * sizeof(int32_t)));
        }
        // populate the first frame
        gpuErrchk(cudaMemcpy(d_great_map, map.data(), _map_X * _map_Y * sizeof(int32_t), cudaMemcpyHostToDevice));
        gpuErrchk(cudaDeviceSynchronize());
        // populate the rest frames
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, populate_great_map, 0, _map_X * _map_Y);
        blockSize = (int)sqrt(blockSize);
        dim3 dimBlock(blockSize, blockSize);
        dim3 dimGrid((_map_X + dimBlock.x - 1) / dimBlock.x, (_map_Y + dimBlock.y - 1) / dimBlock.y);
        populate_great_map<<<dimGrid, dimBlock>>>(d_great_map, _map_X, _map_Y);
        gpuErrchk(cudaDeviceSynchronize());
        if (_verbose) {
            printf("in avg_forest_fire_size\n possible map\n");
            _print_map(d_great_map, _map_X, _map_Y);
        }
        // let's burn the great map and get the results!
        if (ret_fire_size == nullptr) {
            gpuErrchk(cudaMallocManaged(&ret_fire_size, _map_X * _map_Y * sizeof(ssize_t)));
            *ret_fire_size = -1; // it will be set to fire size anyway.
        }
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, fire_size, 0, _map_X * _map_Y);
        blockSize = (int)sqrt(blockSize);
        dim3 dimBlock1(blockSize, blockSize);
        dim3 dimGrid1((_map_X + dimBlock1.x - 1) / dimBlock1.x, (_map_Y + dimBlock1.y - 1) / dimBlock1.y);
        fire_size<<<dimGrid1, dimBlock1>>>(d_great_map, _map_X, _map_Y, ret_fire_size, _verbose);
        gpuErrchk(cudaDeviceSynchronize());

        if (_verbose) {
            printf("in avg_forest_fire_size\n ret_fire_size\n");
            _print_map(ret_fire_size, _map_X, _map_Y);
        }

        // sum up the results: \sum_i s_i p_i
        float avg_fire_size = 0;
        for (ssize_t i = 0; i < _map_X * _map_Y; i++) {
            avg_fire_size += ret_fire_size[i] * _ptr_spark_prob[i];
        }
        return avg_fire_size;
    } else {
        printf("Error: no spark prob data. Please set spark prob first.\n");
        return 0.f;
    }
}