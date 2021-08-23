#include <amgx_c.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include "mpi_test.h"


#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
double second(void)
{
    LARGE_INTEGER t;
    static double oofreq;
    static int checkedForHighResTimer;
    static BOOL hasHighResTimer;

    if (!checkedForHighResTimer) {
        hasHighResTimer = QueryPerformanceFrequency(&t);
        oofreq = 1.0 / (double)t.QuadPart;
        checkedForHighResTimer = 1;
    }
    if (hasHighResTimer) {
        QueryPerformanceCounter(&t);
        return (double)t.QuadPart * oofreq;
    }
    else {
        return (double)GetTickCount() / 1000.0;
    }
}

#elif defined(__linux__) || defined(__QNX__)
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
double second(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#elif defined(__APPLE__)
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/sysctl.h>
double second(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#else
#error unsupported platform
#endif



int main(){ 
    double startSolve, stopSolve;
    //MPI (with CUDA GPUs)
    int rank = 0;
    int lrank = 0;
    int nranks = 0;
    int gpu_count = 0;
    MPI_Comm amgx_mpi_comm = MPI_COMM_WORLD;

    //input matrix and rhs/solution
    int* partition_sizes = NULL;
    int* partition_vector = NULL;
    int partition_vector_size = 0;

    //library handles
    AMGX_Mode mode;
    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;

    //status handling
    AMGX_SOLVE_STATUS status;
    
    /* MPI init (with CUDA GPUs) */
    //MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(amgx_mpi_comm, &nranks);
    MPI_Comm_rank(amgx_mpi_comm, &rank);
    //CUDA GPUs
    CUDA_SAFE_CALL(cudaGetDeviceCount(&gpu_count));
    lrank = rank % gpu_count;
    CUDA_SAFE_CALL(cudaSetDevice(lrank));

    printf("nranks %d gpu conut %d\n", nranks, gpu_count);
    printf("Process %d selecting device %d\n", rank, lrank);

    /* init */
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_initialize_plugins());
    /* system */
    AMGX_SAFE_CALL(AMGX_install_signal_handler());

    mode = AMGX_mode_dDDI;

    AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, "../config/INVERSE_FGMRES"));

    AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));

    AMGX_resources_create(&rsrc, cfg, &amgx_mpi_comm, 1, &lrank);
    AMGX_matrix_create(&A, rsrc, mode);
    AMGX_vector_create(&x, rsrc, mode);
    AMGX_vector_create(&b, rsrc, mode);
    AMGX_solver_create(&solver, rsrc, mode, cfg);

    //read partitioning vector
    partition_vector_size = 406;
    partition_vector = (int*)malloc(partition_vector_size * sizeof(int));
    for (int i = 0; i < partition_vector_size; ++i) {
        partition_vector[i] = i % 4;
    }
    //read the matrix, [and rhs & solution]
    //WARNING: use 1 ring for aggregation and 2 rings for classical path
    int nrings; //=1; //=2;
    AMGX_config_get_default_number_of_rings(cfg, &nrings);
    printf("nrings=%d\n",nrings);

    AMGX_read_system_distributed
    (A, b, x, "../input/A406.tri", nrings, nranks,
        partition_sizes, partition_vector_size, partition_vector);

    //free temporary storage
    if (partition_vector != NULL) { free(partition_vector); }

    startSolve = second();
    /* solver setup */
    //MPI barrier for stability (should be removed in practice to maximize performance)
    //MPI_Barrier(amgx_mpi_comm);
    AMGX_solver_setup(solver, A);
    /* solver solve */
    //MPI barrier for stability (should be removed in practice to maximize performance)
    //MPI_Barrier(amgx_mpi_comm);
    AMGX_solver_solve(solver, b, x);
    /* example of how to change parameters between non-linear iterations */
    //AMGX_config_add_parameters(&cfg, "config_version=2, default:tolerance=1e-12");
    //AMGX_solver_solve(solver, b, x);
    /* example of how to replace coefficients between non-linear iterations */
    //AMGX_matrix_replace_coefficients(A, n, nnz, values, diag);
    //AMGX_solver_setup(solver, A);
    //AMGX_solver_solve(solver, b, x);
    AMGX_solver_get_status(solver, &status);
    stopSolve = second();
    /* example of how to get (the local part of) the solution */
    //int sizeof_v_val;
    //sizeof_v_val = ((NVAMG_GET_MODE_VAL(NVAMG_VecPrecision, mode) == NVAMG_vecDouble))? sizeof(double): sizeof(float);
    //void* result_host = malloc(n*block_dimx*sizeof_v_val);
    //AMGX_vector_download(x, result_host);
    //free(result_host);
    /* destroy resources, matrix, vector and solver */
    AMGX_write_system(A, b, x, "../output/system_mpi.mtx");
    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(rsrc);
    /* destroy config (need to use AMGX_SAFE_CALL after this point) */
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg))
    /* shutdown and exit */
    AMGX_SAFE_CALL(AMGX_finalize_plugins())
    AMGX_SAFE_CALL(AMGX_finalize())
    MPI_Finalize();
    CUDA_SAFE_CALL(cudaDeviceReset());
    printf("Solve time %10.6f sec\n", stopSolve - startSolve);
    return status;
}
int main1(){
    double startSolve, stopSolve;
    //number of outer (non-linear) iterations
    int i = 0;
    int k = 0;
    int max_it = 0;

    //MPI (with CUDA GPUs)
    int rank = 0;
    int lrank = 0;
    int nranks = 0;
    int gpu_count = 0;
    MPI_Comm amgx_mpi_comm = MPI_COMM_WORLD;

    //input matrix and rhs/solution
    int n, nnz, block_dimx, block_dimy, block_size, num_neighbors;
    int* row_ptrs = NULL, * col_indices = NULL, * neighbors = NULL;
    void* values = NULL, * diag = NULL, * dh_x = NULL, * dh_b = NULL;
    int* h_row_ptrs = NULL, * h_col_indices = NULL;
    void* h_values = NULL, * h_diag = NULL, * h_x = NULL, * h_b = NULL;
    int* d_row_ptrs = NULL, * d_col_indices = NULL;
    void* d_values = NULL, * d_diag = NULL, * d_x = NULL, * d_b = NULL;
    int sizeof_m_val;
    int sizeof_v_val;
    int* send_sizes = NULL;
    int** send_maps = NULL;
    int* recv_sizes = NULL;
    int** recv_maps = NULL;
    int* partition_sizes = NULL;
    int* partition_vector = NULL;
    int partition_vector_size = 0;

    //library handles
    AMGX_Mode mode;
    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;

    //status handling
    AMGX_SOLVE_STATUS status;

    /* MPI init (with CUDA GPUs) */
    //MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(amgx_mpi_comm, &nranks);
    MPI_Comm_rank(amgx_mpi_comm, &rank);

    //CUDA GPUs
    CUDA_SAFE_CALL(cudaGetDeviceCount(&gpu_count));
    printf("GPU count %d\n", gpu_count);
    printf("MPI ranks %d\n", nranks);
    lrank = rank % gpu_count;
    CUDA_SAFE_CALL(cudaSetDevice(lrank));
    printf("Process %d selecting device %d\n", rank, lrank);

    /* load the library (if it was dynamically loaded) */
    /* init */
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_initialize_plugins());
    /* system */
    AMGX_SAFE_CALL(AMGX_install_signal_handler());

    mode = AMGX_mode_dDDI;
    sizeof_m_val = ((AMGX_GET_MODE_VAL(AMGX_MatPrecision, mode) == AMGX_matDouble)) ? sizeof(double) : sizeof(float);
    sizeof_v_val = ((AMGX_GET_MODE_VAL(AMGX_VecPrecision, mode) == AMGX_vecDouble)) ? sizeof(double) : sizeof(float);

    /* get max_it number of outer (non-linear) iteration */
    max_it = 2;

    AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, "../config/INVERSE_FGMRES"));
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));
    
    /* create resources, matrix, vector and solver */
    AMGX_resources_create(&rsrc, cfg, &amgx_mpi_comm, 1, &lrank);
    AMGX_matrix_create(&A, rsrc, mode);
    AMGX_vector_create(&x, rsrc, mode);
    AMGX_vector_create(&b, rsrc, mode);
    AMGX_solver_create(&solver, rsrc, mode, cfg);

    /* read and partition the input system: matrix [and rhs & solution]
       Please refer to AMGX_read_system description in the AMGX_Reference.pdf
       manual for details on how to specify the rhs and the solution inside
       the input file. If these are not specified than rhs=[1,...,1]^T and
       (initial guess) sol=[0,...,0]^T. */
       //read partitioning vector
    partition_vector_size = 90449;
    partition_vector = (int*)malloc(partition_vector_size * sizeof(int));
    for (int i = 0; i < partition_vector_size; ++i) {
            partition_vector[i] = i % 128;
    }

    //read the matrix, [and rhs & solution]
    //WARNING: use 1 ring for aggregation path
    int nrings; //=1;
    AMGX_config_get_default_number_of_rings(cfg, &nrings);
    printf("nrings=%d\n",nrings);

    AMGX_read_system_maps_one_ring
    (&n, &nnz, &block_dimx, &block_dimy,
        &h_row_ptrs, &h_col_indices, &h_values, &h_diag, &h_b, &h_x,
        &num_neighbors, &neighbors, &send_sizes, &send_maps, &recv_sizes, &recv_maps,
        rsrc, mode, "../input/s3dkt3m2.mtx", nrings, nranks,
        partition_sizes, partition_vector_size, partition_vector);

    //free temporary storage
    if (partition_vector != NULL) { free(partition_vector); }

    /*
    EXAMPLE
    Say, the initial unpartitioned matrix is:
    CSR row_offsets [0 4 8 13 21 25 32 36 41 46 50 57 61]
    CSR col_indices [0 1 3 8
                     0 1 2 3
                     1 2 3 4 5
                     0 1 2 3 4 5 8 10
                     2 4 5 6
                     2 3 4 5 6 7 10
                     4 5 6 7
                     5 6 7 9 10
                     0 3 8 10 11
                     7 9 10 11
                     3 5 7 8 9 10 11
                     8 9 10 11]
     And we are partitioning it into three pieces with the following partition_vector
     [0 0 0 0 1 1 1 1 2 2 2 2]
     The output of AMGX_read_system_maps_one_ring for partition 0:
     n = 4; nnz = 21;
     row_ptrs = [0 4 8 13 21]
     col_indices = [0 1 3 6
                    0 1 2 3
                    1 2 3 4 5
                    0 1 2 3 4 5 6 7]
     num_neighbors=2; neighbors = [1 2]
     send_sizes = [0 2 4] send_maps = [2 3| 0 3]
     recv_sizes = [0 2 4] recv_maps = [4 5| 6 7]
     global indices mapping to local indices: 0-0 1-1 2-2 3-3 4-4 5-5 8-6 10-7

     The output of AMGX_read_system_maps_one_ring for partition 1:
     n = 4; nnz = 20
     row_ptrs = [0 4 11 15 20]
     col_indices = [4 0 1 2
                    4 5 0 1 2 3 7
                    0 1 2 3
                    1 2 3 6 7]
     num_neighbors=2; neighbors = [0 2]
     send_sizes = [0 2 4] send_maps = [0 1| 1 3]
     recv_sizes = [0 2 4] recv_maps = [4 5| 6 7]
     global indices mapping to local indices: 4-0 5-1 6-2 7-3 2-4 3-5 9-6 10-7

     The output of AMGX_read_system_maps_one_ring for partition 2:
     n = 4; nnz = 20;
     row_ptrs = [0 5 9 16 20]
     col_indices = [4 5 0 2 3
                    7 1 2 3
                    5 6 7 0 1 2 3
                    0 1 2 3]
     num_neighbors=2; neighbors = [0 1]
     send_sizes = [0 2 4] send_maps = [0 2| 1 2]
     recv_sizes = [0 2 4] recv_maps = [4 5| 6 7]
     global indices mapping to local indices: 8-0 9-1 10-2 11-3 0-4 3-5 5-6 7-7
     */
    block_size = block_dimx * block_dimy;
    startSolve = second();
    /* allocate memory and copy the data to the GPU */
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_x, n * block_dimx * sizeof_v_val));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_b, n * block_dimy * sizeof_v_val));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_col_indices, nnz * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_row_ptrs, (n + 1) * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_values, nnz * block_size * sizeof_m_val));

    CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, n * block_dimx * sizeof_v_val, cudaMemcpyDefault));
    CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, n * block_dimy * sizeof_v_val, cudaMemcpyDefault));
    CUDA_SAFE_CALL(cudaMemcpy(d_col_indices, h_col_indices, nnz * sizeof(int), cudaMemcpyDefault));
    CUDA_SAFE_CALL(cudaMemcpy(d_row_ptrs, h_row_ptrs, (n + 1) * sizeof(int), cudaMemcpyDefault));
    CUDA_SAFE_CALL(cudaMemcpy(d_values, h_values, nnz * block_size * sizeof_m_val, cudaMemcpyDefault));

    if (h_diag != NULL)
    {
        CUDA_SAFE_CALL(cudaMalloc(&d_diag, n * block_size * sizeof_m_val));
        CUDA_SAFE_CALL(cudaMemcpy(d_diag, h_diag, n * block_size * sizeof_m_val, cudaMemcpyDefault));
    }

    /* set pointers to point to GPU (device) memory */
    row_ptrs = d_row_ptrs;
    col_indices = d_col_indices;
    values = d_values;
    diag = d_diag;
    dh_x = d_x;
    dh_b = d_b;

    /* set the connectivity information (for the matrix) */
    AMGX_matrix_comm_from_maps_one_ring(A, 1, num_neighbors, neighbors, send_sizes, (const int**)send_maps, recv_sizes, (const int**)recv_maps);
    /* set the connectivity information (for the vector) */
    AMGX_vector_bind(x, A);
    AMGX_vector_bind(b, A);
    /* upload the matrix (and the connectivity information) */
    AMGX_matrix_upload_all(A, n, nnz, block_dimx, block_dimy, row_ptrs, col_indices, values, diag);
    /* upload the vector (and the connectivity information) */
    AMGX_vector_upload(x, n, block_dimx, dh_x);
    AMGX_vector_upload(b, n, block_dimx, dh_b);

    /* start outer (non-linear) iterations */
    for (k = 0; k < max_it; k++)
    {
        /* solver setup */
        //MPI barrier for stability (should be removed in practice to maximize performance)
        //MPI_Barrier(amgx_mpi_comm);
        AMGX_solver_setup(solver, A);
        /* solver solve */
        //MPI barrier for stability (should be removed in practice to maximize performance)
        //MPI_Barrier(amgx_mpi_comm);
        AMGX_solver_solve(solver, b, x);
        /* check the status */
        MPI_Barrier(amgx_mpi_comm);
        AMGX_solver_get_status(solver, &status);

    }

    //AMGX_write_system_distributed(A, b, x, "../output/system_mpi.mtx_distrubed.mtx", nrings, nranks, partition_sizes, partition_vector_size, partition_vector);

    /* deallocate GPU (device) memory */
    CUDA_SAFE_CALL(cudaFree(d_x));
    CUDA_SAFE_CALL(cudaFree(d_b));
    CUDA_SAFE_CALL(cudaFree(d_row_ptrs));
    CUDA_SAFE_CALL(cudaFree(d_col_indices));
    CUDA_SAFE_CALL(cudaFree(d_values));
    stopSolve = second();
    if (d_diag != NULL)
    {
        CUDA_SAFE_CALL(cudaFree(d_diag));
    }

    /* free buffers allocated during AMGX_read_system_maps_one_ring */
    AMGX_free_system_maps_one_ring(h_row_ptrs, h_col_indices, h_values, h_diag, h_b, h_x, num_neighbors, neighbors, send_sizes, send_maps, recv_sizes, recv_maps);
    /* destroy resources, matrix, vector and solver */
    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(rsrc);
    /* destroy config (need to use AMGX_SAFE_CALL after this point) */
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg))
    /* shutdown and exit */
    AMGX_SAFE_CALL(AMGX_finalize_plugins())
    AMGX_SAFE_CALL(AMGX_finalize())
    /* close the library (if it was dynamically loaded) */
    MPI_Finalize();
    CUDA_SAFE_CALL(cudaDeviceReset());
    //return status;
    printf("Solve time %10.6f sec\n", stopSolve - startSolve);
    return 0;
}

