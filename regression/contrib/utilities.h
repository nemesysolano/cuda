/*
 * utilities.h (Formerly Utilities.cuh).
 * Credit to:
 * 		http://stackoverflow.com/questions/22399794/qr-decomposition-to-solve-linear-systems-in-cuda, and
 * 		https://github.com/OrangeOwlSolutions/CUDA-Utilities
 *
 *  Created on: Mar 7, 2017
 *      Author: rsolano
 */

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_device_runtime_api.h>

extern "C" int iDivUp(int, int);
extern "C" void gpuErrchk(cudaError_t);
extern "C" void cusolveSafeCall(cusolverStatus_t);
extern "C" void cublasSafeCall(cublasStatus_t);


#endif /* UTILITIES_H_ */
