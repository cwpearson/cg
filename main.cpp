#include <chrono>
#include <iostream>
#include <random>

#include <Kokkos_Core.hpp>

#include <cusparse_v2.h>

using Clock = std::chrono::steady_clock;
using Dur = std::chrono::duration<double>;


template <typename Scalar>
using Vec = Kokkos::View<Scalar *>;

template <typename Scalar>
struct ELL {
    int64_t rows;
    int64_t cols;
    int64_t nnz;
    int64_t sellValuesSize;
    int64_t sliceSize;

    Vec<int32_t> sellSliceOffsets;
    Vec<int32_t> sellColInd;
    Vec<Scalar> sellValues;


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

template <typename T>
constexpr cudaDataType_t make_cuda_data_type() {
    if constexpr (std::is_same_v<T, float>) {
        return CUDA_R_32F;
    } else if constexpr (std::is_same_v<T, double>) {
        return CUDA_R_64F;
    }else if constexpr (std::is_same_v<T, Kokkos::Experimental::half_t>) {
        return CUDA_R_16F;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return CUDA_R_16BF;
    } else {
        static_assert(std::is_void_v<T> && ! std::is_void_v<T>, "unexpected type");
    }
}

template <typename T>
constexpr cudaDataType_t make_cuda_compute_type() {
    if constexpr (std::is_same_v<T, double>) {
        return CUDA_R_64F;
    } else {
        return CUDA_R_32F;
    }
}

template <typename Scalar>
void spmv(Handle &h, const Vec<Scalar> &y, const ELL<Scalar> &A, const Vec<Scalar> &x) {

    Scalar alpha = 1;
    Scalar beta = 0;

    static bool first = true;
    if (first) {
        // create Y
        cusparseCreateDnVec(&h.y, y.extent(0), y.data(), make_cuda_data_type<Scalar>());

        // create X
        cusparseCreateDnVec(&h.x, x.extent(0), x.data(), make_cuda_data_type<Scalar>());

        // create A
        cusparseIndexType_t sellSliceOffsetsType = CUSPARSE_INDEX_32I;
        cusparseIndexType_t sellColIndType = CUSPARSE_INDEX_32I;
        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
        cudaDataType valueType = make_cuda_data_type<Scalar>();
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
                                make_cuda_compute_type<Scalar>(),
                                CUSPARSE_SPMV_ALG_DEFAULT,
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
             h.A,
             h.x,
             &beta,
             h.y,
             make_cuda_compute_type<Scalar>(),
             CUSPARSE_SPMV_ALG_DEFAULT,
             h.externalBuffer);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": cusparse error: " << cusparseGetErrorString(status) << "\n";
    }

};



template <typename Scalar>
void axpby(const Vec<Scalar> &z, Scalar alpha, const Vec<Scalar> &x, Scalar beta, const Vec<Scalar> &y) {
    Kokkos::parallel_for("axpby", z.extent(0), 
        KOKKOS_LAMBDA(const int i) {
            z(i) = alpha * x(i) + beta * y(i);
        });
}

template <typename Scalar>
Scalar dot(const Vec<Scalar> &x, const Vec<Scalar> &y) {
    Scalar result = 0.0;
    Kokkos::parallel_reduce("dot", x.extent(0), 
        KOKKOS_LAMBDA(const int i, Scalar& lsum) {
            lsum += x(i) * y(i);
        }, result);
    return result;
}

template <typename Scalar>
void dot_ondevice(const Kokkos::View<Scalar> &out, const Vec<Scalar> &x, const Vec<Scalar> &y) {
    Kokkos::parallel_reduce("dot", x.extent(0), 
        KOKKOS_LAMBDA(const int i, Scalar& lsum) {
            lsum += x(i) * y(i);
        }, out);
}

template <typename T>
T sqrt(const T &t) {
    return std::sqrt(t);
}

template<>
__nv_bfloat16 sqrt(const __nv_bfloat16 &t) {
    return __nv_bfloat16(sqrt(float(t)));
}

template<>
Kokkos::Experimental::half_t sqrt(const Kokkos::Experimental::half_t &t) {
    return Kokkos::sqrt(t);
}

template<typename T>
T norm2(const Kokkos::View<T*> &r) {
    return sqrt(dot(r, r));
}

template<typename T>
void norm2_ondevice(Kokkos::View<T> &out, const Kokkos::View<T*> &r) {
    dot_ondevice(out, r, r);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Kokkos::View<T*> &v) {
    constexpr size_t N = 30;  // Default display limit
    size_t n = std::min(N, v.size());
    auto s = Kokkos::subview(v, Kokkos::pair{size_t(0), size_t{n}});
    auto h = Kokkos::create_mirror_view(s);
    Kokkos::deep_copy(h, s);
    os << "[ ";
    for (size_t i = 0; i < n; ++i) {
        if (0 == i) {
            os << double(h(i));
        } else {
            os << ", " << double(h(i));
        }
    }
    if (v.size() > n) {
        os << " ...";
    }
    os << " ]";
    return os;
}

struct Result {
    int iters;
    double error;
    double spmv;
    double reductions;
    double total;
};

template <typename Scalar>
Result cg(const Vec<Scalar>& x, const ELL<Scalar> A, const Vec<Scalar>& b, Scalar tol, bool async = false) {

    std::cout << "A.sellValues=" << A.sellValues << "\n";

    Kokkos::DefaultExecutionSpace space;

    Handle handle;
    cusparseCreate(&handle.h);
    cusparseSetStream(handle.h, space.cuda_stream());

    Vec<Scalar> Ax0("Ax0", A.rows);
    Vec<Scalar> r_k("r", A.rows);
    Vec<Scalar> Ap_k("Ap_k", A.rows);
    Vec<Scalar> r_k1("r_k1", A.rows);
    Vec<Scalar> p_k("p_k", A.rows);

    // sacrificial spmv
    spmv(handle, Ax0, A, x);
    Kokkos::fence();

    auto start = Clock::now();
    // r0 = b - A x0
    spmv(handle, Ax0, A, x);
    if (!async) {
        std::cout << "Ax0=" << Ax0 << "\n";
    }
    axpby(r_k, Scalar{-1}, Ax0, Scalar{1}, b);
    if (!async) {
        std::cout << "r_0=" << r_k << "\n";
    }

    Scalar r = 0;
    Kokkos::View<Scalar> error("error");
    if (async) {
        norm2_ondevice(error, r_k);
    } else {
        r = norm2(r_k);
        std::cerr << __FILE__ << ":" << __LINE__ << " r=" << double(r) << "\n";
        if (r < tol) {
            return Result{0, double(r)};
        }
    }


    Kokkos::deep_copy(space, p_k, r_k); // p0 <- r0

    int k = 1;
    Vec<Scalar> x_k = x;
    for (k = 1; k <= 100; ++k) {
        // std::cerr << __FILE__ << ":" << __LINE__ << " k=" << k << "\n";

        spmv(handle, Ap_k, A, p_k);
        Scalar alpha_k = dot(r_k,r_k) / dot(p_k, Ap_k);
        // std::cerr << __FILE__ << ":" << __LINE__ << " alpha_k=" << alpha_k << "\n";
        
        axpby(r_k1, Scalar{1}, r_k, -alpha_k, Ap_k);

        if (async) {
            norm2_ondevice(error, r_k1);
        } else {
            r = norm2(r_k1);
            std::cerr << __FILE__ << ":" << __LINE__ << " r=" << double(r) << "\n";
            if (r < tol) {
                double elapsed = Dur(Clock::now() - start).count();
                return Result{k-1, double(r), 0, 0, elapsed};
            }
        }

        Scalar beta_k = dot(r_k1, r_k1) / dot(r_k, r_k);
        // std::cerr << __FILE__ << ":" << __LINE__ << " beta_k=" << beta_k << "\n";
        axpby(p_k, Scalar{1}, r_k1, beta_k, p_k);

        axpby(x_k, Scalar{1}, x_k, alpha_k, p_k);
        std::swap(r_k1, r_k);
    }

    if (async) {
        Kokkos::deep_copy(r, error);
        r = sqrt(r);
    } 

    double elapsed = Dur(Clock::now() - start).count();
    return Result{k-1, double(r), 0, 0, elapsed};
}

// cg solve on a region of nx x ny x nz
template <typename Scalar>
Result seven_point(int64_t nx, int64_t ny, int64_t nz, Scalar tol) {
    /*
    for nx x ny x nz gridpoints, the matrix will be (nx*ny*nz) x (nx*ny*nz) with 7 nz in most rows

    neighbors of point i:
    i (self)
    i - 1
    i + 1
    i - nx
    i + nx
    i - nx*ny
    i + nx*ny

    */

    int64_t rows = nx * ny * nz;
    int64_t cols = nx * ny * nz;
    std::cerr << __FILE__ << ":" << __LINE__ << " matrix " << rows << " x " << cols << "\n";


    // basically every row has 7 nnz, so let's just make the whole matrix one slice
    int64_t sliceSize = nx * ny * nz;
    int64_t sellValuesSize = 7 * nx * ny * nz;
    std::cerr << __FILE__ << ":" << __LINE__ << " SELL values =" << sellValuesSize << "\n";

    // actual number of zeros
    std::cerr << __FILE__ << ":" << __LINE__ << " count nnz\n";
    int64_t nnz = 0;
    for (int64_t x = 0; x < nx; ++x) {
        for (int64_t y = 0; y < ny; ++y) {
            for (int64_t z = 0; z < nz; ++z) {
                nnz += 1; // self
                nnz += ((x-1) >= 0);
                nnz += ((x+1) <  nx);
                nnz += ((y-1) >= 0);
                nnz += ((y+1) <  ny);
                nnz += ((z-1) >= 0);
                nnz += ((z+1) <  nz);
            }
        }
    }
    std::cerr << __FILE__ << ":" << __LINE__ << " nnz=" << nnz << "\n";

    std::cerr << __FILE__ << ":" << __LINE__ << " create SELL device views\n";
    Vec<Scalar> sellValues("sellValues", sellValuesSize);
    Vec<int32_t> sellColInd("sellColInd", sellValuesSize);
    Vec<int32_t> sellSliceOffsets("sellSliceOffsets", 1+1); // 1 slice

    std::cerr << __FILE__ << ":" << __LINE__ << " create SELL mirror views\n";
    auto sellValues_h = Kokkos::create_mirror_view(sellValues);
    auto sellColInd_h = Kokkos::create_mirror_view(sellColInd);
    auto sellSliceOffsets_h = Kokkos::create_mirror_view(sellSliceOffsets);

    std::cerr << __FILE__ << ":" << __LINE__ << " fill SELL host views\n";
    sellSliceOffsets_h(0) = 0;
    sellSliceOffsets_h(1) = sellValuesSize;
    for (int64_t i = 0; i < sellValuesSize; ++i) {

        // row of the 2D matrix this value is in
        int64_t row = i % (nx * ny * nz);

        int64_t x = row % nx;
        int64_t y = (row / nx) % ny;
        int64_t z = row / (nx * ny);

        // row entry of the 2D matrix this value represents
        int64_t col_i = i / (nx * ny * nz);

        // the neighbor this column index refers to depends on which column index it is
        // check if the neighbor is inside the 3D grid, and if so, set a value
        // otherwise, set a null column
        switch(col_i) {
            case 3:                   sellValues_h(i) =  6; sellColInd_h(i) = row;                                                              break;
            case 0: if (z - 1 >= 0)  {sellValues_h(i) = -1; sellColInd_h(i) = row - nx * ny;} else {sellColInd_h(i) = -1;} break;
            case 1: if (y - 1 >= 0)  {sellValues_h(i) = -1; sellColInd_h(i) = row - nx     ;} else {sellColInd_h(i) = -1;} break;
            case 2: if (x - 1 >= 0)  {sellValues_h(i) = -1; sellColInd_h(i) = row - 1      ;} else {sellColInd_h(i) = -1;} break;
            case 4: if (x + 1  < nx) {sellValues_h(i) = -1; sellColInd_h(i) = row + 1      ;} else {sellColInd_h(i) = -1;} break;
            case 5: if (y + 1  < ny) {sellValues_h(i) = -1; sellColInd_h(i) = row + nx     ;} else {sellColInd_h(i) = -1;} break;
            case 6: if (z + 1  < nz) {sellValues_h(i) = -1; sellColInd_h(i) = row + nx * ny;} else {sellColInd_h(i) = -1;} break;
            default: ;
        }

#if 0
        std::cerr << "i=" << i 
                  << " row=" << row
                  << " col_i=" << col_i
                  << " edge=" << x << "," << y << "," << z << " -> ";

        switch(col_i) {
            case 3: std::cerr << x << "," << y << "," << z  ; break;
            case 0: std::cerr << x << "," << y << "," << z-1; break;
            case 1: std::cerr << x << "," << y-1 << "," << z; break; 
            case 2: std::cerr << x-1 << "," << y << "," << z; break; 
            case 4: std::cerr << x+1 << "," << y << "," << z; break; 
            case 5: std::cerr << x << "," << y+1 << "," << z; break; 
            case 6: std::cerr << x << "," << y << "," << z+1; break; 
            default: ;
        }

        std::cerr << " sellColInd_h(i)=" << sellColInd_h(i)
                  << " sellValues_h(i)=" << double(sellValues_h(i))
                  << "\n";
#endif
    }

    std::cerr << __FILE__ << ":" << __LINE__ << " sellValues <- sellValues_h\n"; 
    Kokkos::deep_copy(sellValues, sellValues_h);
    std::cerr << __FILE__ << ":" << __LINE__ << " sellColInd <- sellColInd_h\n"; 
    Kokkos::deep_copy(sellColInd, sellColInd_h);
    std::cerr << __FILE__ << ":" << __LINE__ << " sellSliceOffsets <- sellSliceOffsets_h\n"; 
    Kokkos::deep_copy(sellSliceOffsets, sellSliceOffsets_h);

    Vec<Scalar> b("b", cols);
    Vec<Scalar> x("x", rows);

    std::cerr << __FILE__ << ":" << __LINE__ << " b_h = create_mirror_view(b)\n"; 
    auto b_h = Kokkos::create_mirror_view(b);

    std::cerr << __FILE__ << ":" << __LINE__ << " fill b_h\n"; 

    std::mt19937 gen(31337); // an elite choice
    std::uniform_real_distribution<double> dist(0.5, 1.0);
    for (size_t i = 0; i < b_h.extent(0); ++i) {
        b_h(i) = Scalar(dist(gen));
    }

    std::cerr << __FILE__ << ":" << __LINE__ << " b <- b_h\n"; 
    Kokkos::deep_copy(b, b_h);

    std::cout << "b=" << b << "\n";

    ELL<Scalar> A{
        rows, cols, nnz, sellValuesSize, sliceSize, sellSliceOffsets, sellColInd, sellValues
    };

    std::cerr << __FILE__ << ":" << __LINE__ << " cg\n"; 
    return cg(b, A, x, tol);
}



void summarize(const Result &r) {
    std::cout << "cg terminated with k=" << r.iters << " r=" << r.error << std::endl;
    std::cout << "  total:      " << r.total << std::endl;
    std::cout << "  total/iter: " << r.total / r.iters << std::endl;
}


int main(int argc, char** argv) {

    std::string_view dt{"f32"};
    int nx = 10;
    int ny = 10;
    int nz = 10;
    if (argc >= 5) {
        nx = std::atoi(argv[1]);
        nx = std::atoi(argv[2]);
        nx = std::atoi(argv[3]);
        dt = argv[4];
    } else if (argc >= 4) {
        nx = std::atoi(argv[1]);
        nx = std::atoi(argv[2]);
        nx = std::atoi(argv[3]);
    } else if (argc >= 2) {
        nx = ny = nz = std::atoi(argv[1]);
    }

    int err = 0;

    Kokkos::initialize(); {
        if (dt == "f32") {
            summarize(seven_point<float>(nx,ny,nz, 1e-7));
        } else if (dt == "f64") {
            summarize(seven_point<double>(nx,ny,nz, 1e-13));
        } else if (dt == "bf16") {
            summarize(seven_point<__nv_bfloat16>(nx,ny,nz, 1e-3));
        }else if (dt == "f16") {
            summarize(seven_point<Kokkos::Experimental::half_t>(nx,ny,nz, 1e-4));
        } else {
            std::cerr << "unexpected data type " << dt << "\n";
            err = 1;
        }
        
    } Kokkos::finalize();

    return err;
}