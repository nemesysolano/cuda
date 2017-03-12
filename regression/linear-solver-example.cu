#include "linear-solver.h"
#include "contrib/utilities.h"
#include <iostream>
/*
 ============================================================================
 Name        : regression-test.cu
 Author      : Rafael Solano
 Version     :
 Copyright   : (c) 2017
 Description : CUDA compute reciprocals
 ============================================================================
 */
using namespace std;


__global__ void copy_kernel(const double * __restrict d_in1, double * __restrict d_R, const double * __restrict d_C, double * __restrict d_B, const int M, const int N) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < N) && (j < N)) {
        d_R[j * N + i] = d_in1[j * M + i];
        d_B[j * N + i] = d_C[j * M + i];
    }
}

__global__ void identity_kernel(double * d_Q, const int M, const int N) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < N) && (j < N) && (i == j))
    	d_Q[j * N + i] = 1.;
    else
    	d_Q[j * N + i] = 0.;
}

__global__ void init_data(double * d_C, const int M /* Rows*/, const int N /* Columns */) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < N) && (j < N))
    	d_C[j * N + i] = 1.0;
}


int linear_solver_test(int argc, char * argv[]) {

    // --- ASSUMPTION Nrows >= Ncols

    const int Nrows = 7;
    const int Ncols = 5;
    dim3 Grid(iDivUp(Ncols, BLOCK_SIZE), iDivUp(Ncols, BLOCK_SIZE));
    dim3 Block(BLOCK_SIZE, BLOCK_SIZE);

    // --- cuSOLVE input/output parameters/arrays
    int work_size = 0;
    int *devInfo;
    gpuErrchk(cudaMalloc(&devInfo,          sizeof(int)));

    // --- CUDA solver initialization
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    // --- CUBLAS initialization
    cublasHandle_t cublas_handle;
    cublasSafeCall(cublasCreate(&cublas_handle));

    // --- Setting the host, Nrows x Ncols matrix
    double *h_A = (double *)malloc(Nrows * Ncols * sizeof(double));
    for(int row = 0; row < Nrows; row++){
        for(int column = 0; column < Ncols; column++) {
            h_A[row + column*Nrows] = (column + row*row) * sqrt((double)(column + row));
            printf("%8.2f,",h_A[row + column*Nrows]);
        }
        printf("\n");
    }

    // --- Setting the device matrix and moving the host matrix to the device
    double *d_A;
    gpuErrchk(cudaMalloc(&d_A,      Nrows * Ncols * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_A, h_A, Nrows * Ncols * sizeof(double), cudaMemcpyHostToDevice));

    // --- Creates d_Q d_TAU
    double *d_TAU;
    gpuErrchk(cudaMalloc((void**)&d_TAU, min(Nrows, Ncols) * sizeof(double)));

    // --- CUDA QR initialization
    cusolveSafeCall(cusolverDnDgeqrf_bufferSize(solver_handle, Nrows, Ncols, d_A, Nrows, &work_size));

    double *work;
    gpuErrchk(cudaMalloc(&work, work_size * sizeof(double)));

    // --- CUDA GERF execution
    cusolveSafeCall(cusolverDnDgeqrf(solver_handle, Nrows, Ncols, d_A, Nrows, d_TAU, work, work_size, devInfo));
    int devInfo_h = 0;  gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (devInfo_h != 0)
    	std::cout   << "Unsuccessful gerf execution\n\n";

    // --- At this point, the upper triangular part of A contains the elements of R. Showing this.
    gpuErrchk(cudaMemcpy(h_A, d_A, Nrows * Ncols * sizeof(double), cudaMemcpyDeviceToHost));
    for(int j = 0; j < Nrows; j++)
        for(int i = 0; i < Ncols; i++)
            if (i >= j) printf("R[%i, %i] = %f\n", j, i, h_A[j + i*Nrows]);

    // --- Creates d_Q
    double *d_Q;
    gpuErrchk(cudaMalloc(&d_Q,      Nrows * Nrows * sizeof(double)));

    // Initializes d_Q to identity
    identity_kernel<<<Grid, Block>>>(d_Q, Nrows, Ncols);

    // --- CUDA QR execution
    cusolveSafeCall(cusolverDnDormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, Nrows, Ncols, min(Nrows, Ncols), d_A, Nrows, d_TAU, d_Q, Nrows, work, work_size, devInfo));

    // --- Creates d_D
    double *d_D;
    gpuErrchk(cudaMalloc(&d_D,      Nrows  * Nrows  * sizeof(double)));

    // --- Initializes d_D to ones.
    init_data<<<Grid, Block>>>(d_D, Nrows, Nrows);

    // --- CUDA QR execution
    cusolveSafeCall(cusolverDnDormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, Nrows, Ncols, min(Nrows, Ncols), d_A, Nrows, d_TAU, d_D, Nrows, work, work_size, devInfo));

    // --- At this point, d_C contains the elements of Q^T * C, where C is the data vector. Showing this.
    // --- According to the above, only the first column of d_C makes sense.
    //gpuErrchk(cudaMemcpy(h_D, d_D, Nrows * Nrows * sizeof(double), cudaMemcpyDeviceToHost));

    // --- Creates d_R
    double *d_R;
    gpuErrchk(cudaMalloc(&d_R, Ncols * Ncols * sizeof(double)));

    // --- Creates h_B
    double *h_B = (double *)malloc(Ncols * Ncols * sizeof(double));

    // --- Creates d_B
    double *d_B;
    gpuErrchk(cudaMalloc(&d_B, Ncols * Ncols * sizeof(double)));

    // --- Reducing the linear system size
    copy_kernel<<<Grid, Block>>>(d_A, d_R, d_D, d_B, Nrows, Ncols);

    // --- Solving an upper triangular linear system
    const double alpha = 1.;
    cublasSafeCall(cublasDtrsm(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, Ncols, Ncols,
                               &alpha, d_R, Ncols, d_B, Ncols));

    gpuErrchk(cudaMemcpy(h_B, d_B, Ncols * Ncols * sizeof(double), cudaMemcpyDeviceToHost));

    printf("\n\n");
    for (int i=0; i<Ncols; i++){
    	for (int j=0; j<Ncols; j++)
    		printf("%8.3f,",  h_B[i * Ncols + j]);
    	printf("\n");
    }

    free(h_A);
    free(h_B);

    cudaFree(devInfo);
    cudaFree(d_A);
    cudaFree(d_TAU);
    cudaFree(work);
    cudaFree(d_Q);
    cudaFree(d_D);
    cudaFree(d_R);
    cudaFree(d_B);

    cusolverDnDestroy(solver_handle);

    return 0;
}
