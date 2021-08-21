#include <amgx_c.h>


int main()
{
    //versions
    int major, minor;
    char* ver, * date, * time;
    //input matrix and rhs/solution
    int n = 0;
    int bsize_x = 0;
    int bsize_y = 0;
    int sol_size = 0;
    int sol_bsize = 0;
    //library handles
    AMGX_Mode mode;
    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;
    //status handling
    AMGX_SOLVE_STATUS status;

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

    /* create resources, matrix, vector and solver */
    AMGX_resources_create_simple(&rsrc, cfg);
    AMGX_matrix_create(&A, rsrc, mode);
    AMGX_vector_create(&x, rsrc, mode);
    AMGX_vector_create(&b, rsrc, mode);
    AMGX_solver_create(&solver, rsrc, mode, cfg);

    AMGX_read_system(A, b, x, "../input/A.tri");

    AMGX_matrix_get_size(A, &n, &bsize_x, &bsize_y);
    AMGX_vector_get_size(x, &sol_size, &sol_bsize);

    AMGX_vector_set_zero(x, n, bsize_x);

    /* solver setup */
    AMGX_solver_setup(solver, A);
    /* solver solve */
    AMGX_solver_solve(solver, b, x);
    AMGX_solver_get_status(solver, &status);
    AMGX_write_system(A, b, x, "../output/system.mtx");
    /* destroy resources, matrix, vector and solver */
    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(rsrc);
    /* destroy config (need to use AMGX_SAFE_CALL after this point) */
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
    /* shutdown and exit */
    AMGX_SAFE_CALL(AMGX_finalize_plugins());
    AMGX_SAFE_CALL(AMGX_finalize());
    /* close the library (if it was dynamically loaded) */
    return status;
}