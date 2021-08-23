#include "AMGX_helper.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"

double vec_norminf(int n, const double* x)
{
    double norminf = 0;
    for (int j = 0; j < n; j++) {
        double x_abs = fabs(x[j]);
        norminf = (norminf > x_abs) ? norminf : x_abs;
    }
    return norminf;
}
double vec_norm1(int n, const double* x)
{
    double norm1 = 0;
    for (int j = 0; j < n; j++) {
        double x_abs = fabs(x[j]);
        norm1 += x_abs;
    }
    return norm1;
}


void testFidesys(double* result) {
    FILE* fileFid;
    fileFid = fopen("../input/X.vec", "r");
    double* vecFid;
    int sizeFid;
    fscanf(fileFid, "%d", &sizeFid);
    vecFid = (double*)malloc(sizeFid * sizeof(double));
    for (int i = 0; i < sizeFid; ++i) {
        fscanf(fileFid, "%lf", &vecFid[i]);
    }
    printf("inf-norm|Fidesys(X)|             = %e\n", vec_norminf(sizeFid, vecFid));
    printf("inf-norm|AMGX(X)|                = %e\n", vec_norminf(sizeFid, result));
    printf("inf-norm|Fidesys(X)|/|AMGX(X)|   = %lf\n\n", vec_norminf(sizeFid, vecFid) / vec_norminf(sizeFid, result));

    printf("1-norm|Fidesys(X)|               = %e\n", vec_norm1(sizeFid, vecFid));
    printf("1-norm|AMGX(X)|                  = %e\n", vec_norm1(sizeFid, result));
    printf("1-norm|Fidesys(X)|/|AMGX(X)|     = %lf\n\n", vec_norm1(sizeFid, vecFid) / vec_norm1(sizeFid, result));
    for (int i = 0; i < sizeFid; ++i) {
        vecFid[i] -= result[i];
    }

    printf("inf-norm|Fidesys(X) - AMGX(X)|   = %e\n", vec_norminf(sizeFid, vecFid));
    printf("1-norm|Fidesys(X) - AMGX(X)|     = %e\n", vec_norm1(sizeFid, vecFid));
    free(vecFid);
    vecFid = NULL;
    fclose(fileFid);
    fileFid = NULL;
}

void printResult(int n, double* result) {
    FILE* out;
    out = fopen("../output/X_AMGX.vec","w");
    fprintf(out, "%d\n", n);
    for (int i = 0; i < n; ++i) {
        fprintf(out, "%10e\n", result[i]);
    }
    fclose(out);
    out = NULL;
}







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
