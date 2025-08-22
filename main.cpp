#include <iostream>

#include <Kokkos_Core.hpp>

#include <cusparse_v2.h>

// cg solve on a region of N x N x N
void seven_point(int64_t N) {

    /*
    for N x N x N gridpoints, the matrix will be 7N^3 x 7N^3 with 7 nz in most rows
    
    neighbors of point i:
    i (self)
    i - 1
    i + 1
    i - N
    i + N
    i - N^2
    i + N^2
    
    */

    int64_t rows = N * N * N;
    int64_t cols = N * N * N;

    // basically every row has 7 nnz, so let's just make the whole matrix one slice
    int64_t sliceSize = N * N * N;
    int64_t sellValuesSize = 7*N*N*N;

    // actual number of zeros
    int64_t nnz = 0;
    for (int64_t x = 0; x < N; ++x) {
        for (int64_t y = 0; y < N; ++y) {
            for (int64_t z = 0; z < N; ++z) {
                nnz += 1; // self
                nnz += ((x-1) >= 0);
                nnz += ((x+1) <  N);
                nnz += ((y-1) >= 0);
                nnz += ((y+1) <  N);
                nnz += ((z-1) >= 0);
                nnz += ((z+1) <  N);
            }
        }
    }

    Kokkos::View<float*> sellValues("sellValues", sellValuesSize);
    Kokkos::View<int32_t*> sellColInd("sellColInd", sellValuesSize);
    Kokkos::View<int32_t*> sellSliceOffsets("sellSliceOffsets", 1+1); // 1 slice

    auto sellValues_h = Kokkos::create_mirror_view(sellValues);
    auto sellColInd_h = Kokkos::create_mirror_view(sellColInd);
    auto sellSliceOffsets_h = Kokkos::create_mirror_view(sellSliceOffsets);

    sellSliceOffsets_h(0) = 0;
    sellSliceOffsets_h(1) = sellValuesSize;

    for (int64_t i = 0; i < sellValuesSize; ++i) {

        // row of the 2D matrix this value is in
        int64_t row = i % (N * N * N);
        
        int64_t x = row % N;
        int64_t y = (row / N) % N;
        int64_t z = row / N / N;
        
        // row entry of the 2D matrix this value represents
        int64_t col_i = i / (N * N * N);
        
        // the neighbor this column index refers to depends on which column index it is
        // check if the neighbor is inside the 3D grid, and if so, set a value
        // otherwise, set a null column
        switch(col_i) {
            case 3:                  sellValues_h(i) = 0.5; sellColInd_h(i) = row;                                                       break;
            case 0: if (z - 1 >= 0) {sellValues_h(i) = 0.5; sellColInd_h(i) = row - N * N;} else {sellColInd_h(i) = -1;} break;
            case 1: if (y - 1 >= 0) {sellValues_h(i) = 0.5; sellColInd_h(i) = row - N    ;} else {sellColInd_h(i) = -1;} break;
            case 2: if (x - 1 >= 0) {sellValues_h(i) = 0.5; sellColInd_h(i) = row - 1    ;} else {sellColInd_h(i) = -1;} break;
            case 4: if (x + 1  < N) {sellValues_h(i) = 0.5; sellColInd_h(i) = row + 1    ;} else {sellColInd_h(i) = -1;} break;
            case 5: if (y + 1  < N) {sellValues_h(i) = 0.5; sellColInd_h(i) = row + N    ;} else {sellColInd_h(i) = -1;} break;
            case 6: if (z + 1  < N) {sellValues_h(i) = 0.5; sellColInd_h(i) = row + N * N;} else {sellColInd_h(i) = -1;} break;
            default: ;
        }

        // std::cerr << "i=" << i 
        //           << " row=" << row
        //           << " col_i=" << col_i
        //           << " sellColInd_h(i)=" << sellColInd_h(i)
        //           << "\n";
    }

    Kokkos::deep_copy(sellValues, sellValues_h);
    Kokkos::deep_copy(sellColInd, sellColInd_h);

    cusparseConstSpMatDescr_t spMatDescr;
    cusparseIndexType_t sellSliceOffsetsType = CUSPARSE_INDEX_32I;
    cusparseIndexType_t sellColIndType = CUSPARSE_INDEX_32I;
    cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
    cudaDataType valueType = CUDA_R_32F;
    cusparseStatus_t status = cusparseCreateConstSlicedEll(&spMatDescr, rows, cols, nnz,
                             sellValuesSize,
                             sliceSize,
                             sellSliceOffsets.data(),
                             sellColInd.data(),
                             sellValues.data(),
                             sellSliceOffsetsType,
                             sellColIndType,
                             idxBase,
                             valueType);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": cusparse error: " << cusparseGetErrorString(status) << "\n";
    }


    Kokkos::View<float *> y("y", cols);
    Kokkos::View<float *> x("x", rows);

    cusparseDnVecDescr_t xDesc;
    cusparseCreateDnVec(&xDesc, x.extent(0), x.data(), CUDA_R_32F);

    cusparseDnVecDescr_t yDesc;
    cusparseCreateDnVec(&yDesc, y.extent(0), y.data(), CUDA_R_32F);

cusparseHandle_t handle;
status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": cusparse error: " << cusparseGetErrorString(status) << "\n";
    }
// TODO: init handle

float alpha = 1;
float beta = 0;
size_t bufferSize;
status = cusparseSpMV_bufferSize(handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha,
                        spMatDescr,
                        xDesc,
                        &beta,
                        yDesc,
                        CUDA_R_32F,
                        CUSPARSE_SPMV_SELL_ALG1,
                        &bufferSize);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": cusparse error: " << cusparseGetErrorString(status) << "\n";
    }
    std::cerr << bufferSize << "\n";

    // TODO: allocate buffer
    if (bufferSize != 0) {
        throw std::runtime_error("need to allocate buffer");
    }
    void *externalBuffer = nullptr;


status = cusparseSpMV(handle,
             CUSPARSE_OPERATION_NON_TRANSPOSE,
             &alpha,
             spMatDescr,  // non-const descriptor supported
             xDesc,  // non-const descriptor supported
             &beta,
             yDesc,
             CUDA_R_32F,
             CUSPARSE_SPMV_SELL_ALG1,
             externalBuffer);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": cusparse error: " << cusparseGetErrorString(status) << "\n";
    }

    cusparseDestroy(handle);
}


int main(void) {
    Kokkos::initialize(); {
        seven_point(4);
    } Kokkos::finalize();
}