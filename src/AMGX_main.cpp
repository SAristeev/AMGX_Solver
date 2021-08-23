#include <cuda_runtime.h>
#include <amgx_c.h>
#include "AMGX_helper.h"

int main(){
    double startInput, stopInput, startSolve, stopSolve,startInit, stopInit, startDestroy, stopDestroy;
    //versions
    int major, minor;
    char* ver, * date, * time;
    //input matrix and rhs/solution
    int n = 0;
    int bsize_x = 0;
    int bsize_y = 0;
    int sol_size = 0;
    int sol_bsize = 0;
    
    double* result;
    //library handles
    AMGX_Mode mode;
    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;
    //status handling
    AMGX_SOLVE_STATUS status;


    startInit = second();
    /* init */
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_initialize_plugins());
    /* system */
    AMGX_SAFE_CALL(AMGX_install_signal_handler());

    AMGX_get_api_version(&major, &minor);
    printf("amgx api version: %d.%d\n", major, minor);
    AMGX_get_build_info_strings(&ver, &date, &time);
    printf("amgx build version: %s\nBuild date and time: %s %s\n", ver, date, time);

    mode = AMGX_mode_dDDI;
    AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, "../config/JACOBI_DAVIDSON"));
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));

    AMGX_resources_create_simple(&rsrc, cfg);
    AMGX_matrix_create(&A, rsrc, mode);
    AMGX_vector_create(&x, rsrc, mode);
    AMGX_vector_create(&b, rsrc, mode);
    AMGX_solver_create(&solver, rsrc, mode, cfg);
    stopInit = second();
    startInput = second();
    AMGX_read_system(A, b, x, "../input/A406.tri");
    stopInput = second();

    startSolve = second();

    AMGX_matrix_get_size(A, &n, &bsize_x, &bsize_y);
    AMGX_vector_get_size(x, &sol_size, &sol_bsize);
    AMGX_vector_set_zero(x, n, bsize_x);

    AMGX_solver_setup(solver, A);
    AMGX_solver_solve(solver, b, x);
    AMGX_solver_get_status(solver, &status);
    stopSolve = second();
    result = (double*)malloc(sol_size * sol_bsize * sizeof(double));
    
    AMGX_pin_memory(result, sol_size * sol_bsize * sizeof(double));
    AMGX_vector_download(x, result);
    testFidesys(result);
    printResult(sol_size * sol_bsize, result);
    startDestroy = second();
    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
        
    AMGX_resources_destroy(rsrc);
        
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
    AMGX_SAFE_CALL(AMGX_finalize_plugins());
    AMGX_SAFE_CALL(AMGX_finalize());
    stopDestroy = second();
    

    free(result);
    result = NULL;
    printf("Initialize time    %10.6f sec\n", stopInit - startInit);
    printf("Input time         %10.6f sec\n", stopInput - startInput);
    printf("Solve time         %10.6f sec\n", stopSolve - startSolve);
    printf("Destroy time       %10.6f sec\n", stopDestroy - startDestroy);
return 0;
}