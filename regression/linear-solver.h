/*
 * linear-solver.h
 *
 *  Created on: Mar 7, 2017
 *      Author: rsolano
 */

#ifndef LINEAR_SOLVER_H_
#define LINEAR_SOLVER_H_
#include <matrix.h>
#include <string>
#include <stdexcept>
#include "contrib/utilities.h"

#define BLOCK_SIZE 32 //TODO: Calculate BLOCK_SIZE according to GPU capabilities.

int linear_solver_test(int argc, char * argv[]);
namespace regression {
	/** Error code ranging between 0 and 100 are reserved for exceptions thrown from IndicatorError's methods.
	 *
	 */
	enum LINEAR_SOLVER_ERROR_ENUM {
		MORE_COLUMNS_THAN_ROWS,
		EMPTY_OPERANDS,
		COEFFICIENT_MATRIX_COLS_MISMATCH,
		ROWS_MISMATCH
	};
	/** This exception class reports constraint violations detected in data provided to linear solvers.
	 *
	 */
	class LinearSolverError: public std::runtime_error {
	protected:
		int code = 0 /**< Custom error message. */;
	public:
		/** calls runtime_error(const string &) and initializes this->code;
		 *
		 */
		 inline explicit LinearSolverError (int code_ /**< Custom error code. */, const std::string& what_arg /**< Custom error message. */): runtime_error(what_arg), code(code_) {

		 }

		 /** calls runtime_error(const char *) and initializes this->code;
		  *
		  */
		 inline explicit LinearSolverError (int code_ /**< Custom error code. */, const char* what_arg /**< Custom error message. */): runtime_error(what_arg), code(code_) {

		 }

		 /**
		  * @return this->code
		  */
		 inline int Code() { return code; }
	};



	/** This class represents a linear (square) system. A*X=B
	 *
	 */
	class LinearSolver {
		double *d_A = nullptr;  /**< Device buffer for coefficients */
		double *d_TAU= nullptr; /**< Device buffer for τ */
		double *work= nullptr; /**< Device intermediate buffer. */
		double *d_Q= nullptr; /**< Device buffer for CUDA QR execution. */
		double *d_D= nullptr; /**< Device buffer for original/non-reduced right side matrix. */
		double *d_R= nullptr; /**< Device buffer reduced right side matrix. */
		double *d_B= nullptr; /**< Device buffer system's solution. */
		unsigned rows = 0; /** Row count (rows >= columns)*/
		unsigned columns = 0; /** Column count */
	    dim3 * Grid= nullptr;
	    dim3 * Block= nullptr;
	    int work_size = 0;
	    int *devInfo = nullptr; /**< device info pointer*/
	    cusolverDnHandle_t solver_handle = nullptr; /**< solver handle */
	    cublasHandle_t cublas_handle = nullptr;/**< cublas handle */
	public:
	    /** Copy constructor */
		LinearSolver (const LinearSolver& other /**< The source matrix */) = delete;

	    /** Move constructor */
		LinearSolver (LinearSolver&& other /**< The source matrix */) = delete;

		/** Copy operator */
		LinearSolver& operator = (const LinearSolver & other /**< The source matrix */) = delete;

		/** Move assignment operator */
		LinearSolver& operator = (LinearSolver && other /**< The source matrix */) = delete;


		/** Creates the internal GPU & Host buffers.
		 *  @throws LinearSystemException if there is not enough GPU RAM to create internal buffers.
		 */
		LinearSolver(
			unsigned rows_ /**< [in] Strict positive integer. Must be >= columns_ */,
			unsigned columns_ /**< [in] Strict positive integer.  */
		);

		/** This () operator converts this class into a callable.
		 *
		 */
		matrix::Matrix operator ()  (
			matrix::Matrix & A /**< Coefficients Matrix */,
			matrix::Matrix & B /**< Right side matriz */
		);

		virtual ~LinearSolver();
	};


}



#endif /* LINEAR_SOLVER_H_ */
