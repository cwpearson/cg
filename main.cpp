#include <iostream>
#include <random>

#include <Kokkos_Core.hpp>

#include <cusparse_v2.h>

using Vec = Kokkos::View<float*>;

struct ELL {
    int64_t rows;
    int64_t cols;
    int64_t nnz;
    int64_t sellValuesSize;
    int64_t sliceSize;

    Kokkos::View<int32_t*> sellSliceOffsets;
    Kokkos::View<int32_t*> sellColInd;
    Vec sellValues;


};

struct Handle {
    cusparseHandle_t h;
    cusparseConstSpMatDescr_t A;
    cusparseDnVecDescr_t x;
    cusparseDnVecDescr_t y;
    void *externalBuffer;

    ~Handle() {
        cusparseDestroy(h);
    }
};



void spmv(Handle &h, const Vec &y, const ELL &A, const Vec &x) {

    constexpr float alpha = 1;
    constexpr float beta = 0;

    static bool first = true;
    if (first) {
        cusparseCreate(&h.h);
        cusparseSetStream(h.h, Kokkos::DefaultExecutionSpace{}.cuda_stream());

        // create Y
        cusparseCreateDnVec(&h.y, y.extent(0), y.data(), CUDA_R_32F);

        // create X
        cusparseCreateDnVec(&h.x, x.extent(0), x.data(), CUDA_R_32F);

        // create A
        cusparseIndexType_t sellSliceOffsetsType = CUSPARSE_INDEX_32I;
        cusparseIndexType_t sellColIndType = CUSPARSE_INDEX_32I;
        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
        cudaDataType valueType = CUDA_R_32F;
        cusparseStatus_t status = cusparseCreateConstSlicedEll(&h.A, A.rows, A.cols, A.nnz,
                             A.sellValuesSize,
                             A.sliceSize,
                             A.sellSliceOffsets.data(),
                             A.sellColInd.data(),
                             A.sellValues.data(),
                             sellSliceOffsetsType,
                             sellColIndType,
                             idxBase,
                             valueType);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << __FILE__ << ":" << __LINE__ << ": cusparse error: " << cusparseGetErrorString(status) << "\n";
        }

        // compute the buffer size
        size_t bufferSize;
        status = cusparseSpMV_bufferSize(h.h,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha,
                                h.A,
                                h.x,
                                &beta,
                                h.y,
                                CUDA_R_32F,
                                CUSPARSE_SPMV_SELL_ALG1,
                                &bufferSize);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << __FILE__ << ":" << __LINE__ << ": cusparse error: " << cusparseGetErrorString(status) << "\n";
        }
        // TODO: allocate buffer
        if (bufferSize != 0) {
            throw std::runtime_error("need to allocate buffer");
        }
        h.externalBuffer = nullptr;

        first = false;
    }

    // update where x and y point
    cusparseDnVecSetValues(h.x, x.data());
    cusparseDnVecSetValues(h.y, y.data());

    cusparseStatus_t status = cusparseSpMV(h.h,
             CUSPARSE_OPERATION_NON_TRANSPOSE,
             &alpha,
             h.A,  // non-const descriptor supported
             h.x,  // non-const descriptor supported
             &beta,
             h.y,
             CUDA_R_32F,
             CUSPARSE_SPMV_SELL_ALG1,
             h.externalBuffer);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": cusparse error: " << cusparseGetErrorString(status) << "\n";
    }

};



void axpby(Vec &z, float alpha, const Vec &x, float beta, const Vec &y) {
    Kokkos::parallel_for("axpby", z.extent(0), 
        KOKKOS_LAMBDA(const int i) {
            z(i) = alpha * x(i) + beta * y(i);
        });
}

float dot(const Vec &x, const Vec &y) {
    float result = 0.0;
    Kokkos::parallel_reduce("dot", x.extent(0), 
        KOKKOS_LAMBDA(const int i, float& lsum) {
            lsum += x(i) * y(i);
        }, result);
    return result;
}

float norm2(const Vec &r) {
    return std::sqrt(dot(r, r));
}

std::pair<int, float> cg(const Vec& x, const ELL A, const Vec& b, float tol) {

    Handle handle;

    Vec Ax0("Ax0", A.rows);
    Vec r_k("r", A.rows);

    // r0 = b - A x0
    spmv(handle, Ax0, A, x);
    axpby(r_k, -1, Ax0, 1, b);

    float r = norm2(r_k);
    std::cerr << __FILE__ << ":" << __LINE__ << " r=" << r << "\n";
    if (r < tol) {
        return std::make_pair(0, r);
    }

    Vec p_k("p_k", A.rows);
    Kokkos::deep_copy(p_k, r_k); // p0 <- r0

    Vec x_k = x;
    Vec Ap_k("Ap_k", A.rows);
    Vec r_k1("r_k1", A.rows);
    int k = 1;
    for (k = 1; k <= 1000; ++k) {
        // std::cerr << __FILE__ << ":" << __LINE__ << " k=" << k << "\n";

        spmv(handle, Ap_k, A, p_k);
        float alpha_k = dot(r_k,r_k) / dot(p_k, Ap_k);
        // std::cerr << __FILE__ << ":" << __LINE__ << " alpha_k=" << alpha_k << "\n";
        
        axpby(r_k1, 1, r_k, -alpha_k, Ap_k);

        r = norm2(r_k1);
        std::cerr << __FILE__ << ":" << __LINE__ << " r=" << r << "\n";
        if (r < tol) {
            return std::make_pair(k, r);
        }

        float beta_k = dot(r_k1, r_k1) / dot(r_k, r_k);
        // std::cerr << __FILE__ << ":" << __LINE__ << " beta_k=" << beta_k << "\n";
        axpby(p_k, 1, r_k1, beta_k, p_k);


        axpby(x_k, 1, x_k, alpha_k, p_k);
        std::swap(r_k1, r_k);
    }

    return std::make_pair(k-1, r);
}

// cg solve on a region of N x N x N
void seven_point(int64_t N) {

    std::mt19937 gen(31337); // an elite choice
    std::uniform_real_distribution<float> dist(0.5, 1.0);

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
    std::cerr << __FILE__ << ":" << __LINE__ << " matrix " << rows << " x " << cols << "\n";


    // basically every row has 7 nnz, so let's just make the whole matrix one slice
    int64_t sliceSize = N * N * N;
    int64_t sellValuesSize = 7*N*N*N;
    std::cerr << __FILE__ << ":" << __LINE__ << " SELL values =" << sellValuesSize << "\n";

    // actual number of zeros
    std::cerr << __FILE__ << ":" << __LINE__ << " count nnz\n";
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
    std::cerr << __FILE__ << ":" << __LINE__ << " nnz=" << nnz << "\n";

    std::cerr << __FILE__ << ":" << __LINE__ << " create SELL device views\n";
    Kokkos::View<float*> sellValues("sellValues", sellValuesSize);
    Kokkos::View<int32_t*> sellColInd("sellColInd", sellValuesSize);
    Kokkos::View<int32_t*> sellSliceOffsets("sellSliceOffsets", 1+1); // 1 slice

    std::cerr << __FILE__ << ":" << __LINE__ << " create SELL mirror views\n";
    auto sellValues_h = Kokkos::create_mirror_view(sellValues);
    auto sellColInd_h = Kokkos::create_mirror_view(sellColInd);
    auto sellSliceOffsets_h = Kokkos::create_mirror_view(sellSliceOffsets);

    std::cerr << __FILE__ << ":" << __LINE__ << " fill SELL host views\n";
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
            case 3:                  sellValues_h(i) =  6; sellColInd_h(i) = row;                                                       break;
            case 0: if (z - 1 >= 0) {sellValues_h(i) = -1; sellColInd_h(i) = row - N * N;} else {sellColInd_h(i) = -1;} break;
            case 1: if (y - 1 >= 0) {sellValues_h(i) = -1; sellColInd_h(i) = row - N    ;} else {sellColInd_h(i) = -1;} break;
            case 2: if (x - 1 >= 0) {sellValues_h(i) = -1; sellColInd_h(i) = row - 1    ;} else {sellColInd_h(i) = -1;} break;
            case 4: if (x + 1  < N) {sellValues_h(i) = -1; sellColInd_h(i) = row + 1    ;} else {sellColInd_h(i) = -1;} break;
            case 5: if (y + 1  < N) {sellValues_h(i) = -1; sellColInd_h(i) = row + N    ;} else {sellColInd_h(i) = -1;} break;
            case 6: if (z + 1  < N) {sellValues_h(i) = -1; sellColInd_h(i) = row + N * N;} else {sellColInd_h(i) = -1;} break;
            default: ;
        }

        // std::cerr << "i=" << i 
        //           << " row=" << row
        //           << " col_i=" << col_i
        //           << " edge=" << x << "," << y << "," << z << " -> ";

        // switch(col_i) {
        //     case 3: std::cerr << x << "," << y << "," << z  ; break;
        //     case 0: std::cerr << x << "," << y << "," << z-1; break;
        //     case 1: std::cerr << x << "," << y-1 << "," << z; break; 
        //     case 2: std::cerr << x-1 << "," << y << "," << z; break; 
        //     case 4: std::cerr << x+1 << "," << y << "," << z; break; 
        //     case 5: std::cerr << x << "," << y+1 << "," << z; break; 
        //     case 6: std::cerr << x << "," << y << "," << z+1; break; 
        //     default: ;
        // }

        // std::cerr << " sellColInd_h(i)=" << sellColInd_h(i)
        //           << " sellValues_h(i)=" << sellValues_h(i)
        //           << "\n";
    }

    std::cerr << __FILE__ << ":" << __LINE__ << " sellValues <- sellValues_h\n"; 
    Kokkos::deep_copy(sellValues, sellValues_h);
    std::cerr << __FILE__ << ":" << __LINE__ << " sellColInd <- sellColInd_h\n"; 
    Kokkos::deep_copy(sellColInd, sellColInd_h);
    std::cerr << __FILE__ << ":" << __LINE__ << " sellSliceOffsets <- sellSliceOffsets_h\n"; 
    Kokkos::deep_copy(sellSliceOffsets, sellSliceOffsets_h);

    Kokkos::View<float *> b("b", cols);
    Kokkos::View<float *> x("x", rows);

    std::cerr << __FILE__ << ":" << __LINE__ << " b_h = create_mirror_view(b)\n"; 
    auto b_h = Kokkos::create_mirror_view(b);

    std::cerr << __FILE__ << ":" << __LINE__ << " fill b_h\n"; 

    for (size_t i = 0; i < b_h.extent(0); ++i) {
        b_h(i) = dist(gen);
    }

    std::cerr << __FILE__ << ":" << __LINE__ << " b <- b_h\n"; 
    Kokkos::deep_copy(b, b_h);

    ELL A{
        rows, cols, nnz, sellValuesSize, sliceSize, sellSliceOffsets, sellColInd, sellValues
    };

    std::cerr << __FILE__ << ":" << __LINE__ << " cg\n"; 
    const auto p = cg(b, A, x, 1e-7);

    std::cerr << __FILE__ << ":" << __LINE__ << " cg terminated with k=" << p.first << " r=" << p.second << "\n";

}




int main(int argc, char** argv) {

    int N = 10;
    if (argc >= 2) {
        N = std::atoi(argv[1]);
    }

    Kokkos::initialize(); {
        seven_point(N);
    } Kokkos::finalize();

    return 0;
}