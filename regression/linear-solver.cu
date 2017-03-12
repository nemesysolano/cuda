#include "linear-solver.h"
#include "contrib/utilities.h"
#include <iostream>
#include <string>
//TODO: In order to provide a fair service to all processes without exhausting GPU resources, we are creating an object pool to service instances of this class

// http://stackoverflow.com/questions/27827923/c-object-pool-that-provides-items-as-smart-pointers-that-are-returned-to-pool
using namespace std;
using namespace regression;
using namespace matrix;

static const char* LINEAR_SOLVER_ERROR_MESSAGES[] = {
	"@LinearSolver::LinearSolver(rows_, columns_). columns_ > rows_ ",
	"@LinearSolver::operator(matrix::Matrix & A, matrix::Matrix & B). Either A or B is empty.",
	"@LinearSolver::operator(matrix::Matrix & A, matrix::Matrix & B). Coefficient matrix's column count is not equals solver's column count.",
	"@LinearSolver::operator(matrix::Matrix & A, matrix::Matrix & B). A.Rows() > this->rows || B.Rows() > this->rows || A.Rows() != B.Rows()"
};

__global__ void internal_copy_kernel(const double * __restrict d_in1, double * __restrict d_R, const double * __restrict d_C, double * __restrict d_B, const int M, const int N) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < N) && (j < N)) {
        d_R[j * N + i] = d_in1[j * M + i];
        d_B[j * N + i] = d_C[j * M + i];
    }
}

__global__ void internal_identity_kernel(double * d_Q, const int M, const int N) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < N) && (j < N) && (i == j))
    	d_Q[j * N + i] = 1.;
    else
    	d_Q[j * N + i] = 0.;
}

__global__ void internal_init_data(double * d_C, const int M /* Rows*/, const int N /* Columns */) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < N) && (j < N))
    	d_C[j * N + i] = 1.0;
}

/** Creates the internal GPU & Host buffers.
 *  @throws LinearSolverError if columns_ > rows_
 */
LinearSolver::LinearSolver(
	unsigned rows_ /**< [in] Strict positive integer. Must be >= columns_ */,
	unsigned columns_ /**< [in] Strict positive integer.  */
):rows(rows_), columns(columns_) {
	size_t Nrows = rows;
	size_t Ncols = columns;

	if(columns > rows)
		throw LinearSolverError((int)MORE_COLUMNS_THAN_ROWS, LINEAR_SOLVER_ERROR_MESSAGES[(int)MORE_COLUMNS_THAN_ROWS]);

    Grid = new dim3(iDivUp(columns, BLOCK_SIZE), iDivUp(columns, BLOCK_SIZE));
    Block = new dim3(BLOCK_SIZE, BLOCK_SIZE);

    // --- cuSOLVE input/output parameters/arrays
    gpuErrchk(cudaMalloc(&devInfo, sizeof(int)));

    // --- CUDA solver initialization
    cusolverDnCreate(&solver_handle);

    // --- CUBLAS initialization
    cublasSafeCall(cublasCreate(&cublas_handle));

    // --- Setting the device matrix and moving the host matrix to the device
    gpuErrchk(cudaMalloc((void**)&d_A,  Nrows * Ncols * sizeof(double)));

    // --- Creates d_Q d_TAU
    gpuErrchk(cudaMalloc((void**)&d_TAU, min(Nrows, Ncols) * sizeof(double)));

    // --- CUDA QR initialization
    cusolveSafeCall(cusolverDnDgeqrf_bufferSize(solver_handle, Nrows, Ncols, d_A, Nrows, &work_size));
    gpuErrchk(cudaMalloc((void**)&work, work_size * sizeof(double)));

    // --- Creates d_Q
    gpuErrchk(cudaMalloc((void**)&d_Q, Nrows * Nrows * sizeof(double)));

    // --- Creates d_D
    gpuErrchk(cudaMalloc((void**)&d_D, Nrows  * Nrows  * sizeof(double)));

    // --- Creates d_R
    gpuErrchk(cudaMalloc((void**)&d_R, Ncols * Ncols * sizeof(double)));

    // --- Creates d_B
    gpuErrchk(cudaMalloc((void**)&d_B, Ncols * Ncols * sizeof(double)));

}

/** This () operator converts this class into a callable.
 *
 */
matrix::Matrix LinearSolver::operator () (
	matrix::Matrix & A /**< Coefficient Matrix */,
	matrix::Matrix & B /**< Right side matriz */
) {

    const int Nrows = (int)rows;
    const int Ncols = (int)columns;

	if(A.IsEmpty() || B.IsEmpty())
		throw LinearSolverError((int)EMPTY_OPERANDS, LINEAR_SOLVER_ERROR_MESSAGES[(int)EMPTY_OPERANDS]);

	if(A.Columns() != columns)
		throw LinearSolverError((int)COEFFICIENT_MATRIX_COLS_MISMATCH, LINEAR_SOLVER_ERROR_MESSAGES[(int)COEFFICIENT_MATRIX_COLS_MISMATCH]);

	if(A.Rows() != rows || B.Rows() != rows || A.Rows() != B.Rows())
		throw LinearSolverError((int)ROWS_MISMATCH, LINEAR_SOLVER_ERROR_MESSAGES[(int)ROWS_MISMATCH]);

	 // --- Setting the device matrix and moving the host matrix to the device
	const double * h_A = A.Buffer();
	gpuErrchk(cudaMemcpy(d_A, h_A, Nrows * Ncols * sizeof(double), cudaMemcpyHostToDevice));

   // --- CUDA GERF execution
	cusolveSafeCall(cusolverDnDgeqrf(solver_handle, Nrows, Ncols, d_A, Nrows, d_TAU, work, work_size, devInfo));
	int devInfo_h = 0;
	gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));

	if (devInfo_h != 0)
		std::cout   << "Unsuccessful gerf execution\n\n";

    // --- At this point, the upper triangular part of A contains the elements of R. Showing this.
    gpuErrchk(cudaMemcpy(( double *)h_A, d_A, Nrows * Ncols * sizeof(double), cudaMemcpyDeviceToHost));

    // Initializes d_Q to identity
    internal_identity_kernel<<<*Grid, *Block>>>(d_Q, Nrows, Ncols);

    // --- CUDA QR execution
    cusolveSafeCall(cusolverDnDormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, Nrows, Ncols, min(Nrows, Ncols), d_A, Nrows, d_TAU, d_Q, Nrows, work, work_size, devInfo));

    // --- Initializes d_D to ones.
    internal_init_data<<<*Grid, *Block>>>(d_D, Nrows, Nrows);

    // --- CUDA QR execution
    cusolveSafeCall(cusolverDnDormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, Nrows, Ncols, min(Nrows, Ncols), d_A, Nrows, d_TAU, d_D, Nrows, work, work_size, devInfo));

    // --- Creates h_B
    Matrix cublasDstrmOutput(Ncols, Ncols);
    double *h_B = (double *)cublasDstrmOutput.Buffer();

    // --- Reducing the linear system size
    internal_copy_kernel<<<*Grid, *Block>>>(d_A, d_R, d_D, d_B, Nrows, Ncols);

    // --- Solving an upper triangular linear system
    const double alpha = 1.;
    cublasSafeCall(cublasDtrsm(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, Ncols, Ncols,
                               &alpha, d_R, Ncols, d_B, Ncols));

    gpuErrchk(cudaMemcpy(h_B, d_B, Ncols * Ncols * sizeof(double), cudaMemcpyDeviceToHost));

	return cublasDstrmOutput;
}

LinearSolver::~LinearSolver() {
	delete Grid;
	delete Block;
	cudaFree(devInfo);
	cusolverDnDestroy(solver_handle);
	cublasDestroy(cublas_handle);
	cudaFree(d_A);
	cudaFree(d_TAU);
	cudaFree(work);
	cudaFree(d_Q);
	cudaFree(d_D);
	cudaFree(d_R);
	cudaFree(d_B);

}
