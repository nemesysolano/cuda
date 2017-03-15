/*
 * matrix.h
 *
 *  Created on: Mar 8, 2017
 *      Author: rsolano
 */

#ifndef MATRIX_H_
#define MATRIX_H_
#include "matrix.h"
#include <cstdlib>
#include <iostream>
namespace matrix {
	/** Error code ranging between 0 and 100 are reserved for exceptions thrown from Matrix'S methods.
	 *
	 */
	enum MATRIX_ERROR_ENUM {
		UNEXPECTED_EOF_IN_CONSTRUCTOR,
		BAD_INPUT_STREAM
	};

	extern const char* MATRIX_ERROR_MESSAGES[];

	/** This exception reports issues that may occur inside Matrix class' methods.
	 *
	 */
	class MatrixError: public std::runtime_error {
	protected:
		int code = 0 /**< Custom error message. */;
	public:
		/** calls runtime_error(const string &) and initializes this->code;
		 *
		 */
		 inline explicit MatrixError (int code_ /**< Custom error code. */, const std::string& what_arg /**< Custom error message. */): runtime_error(what_arg), code(code_) {

		 }

		 /** calls runtime_error(const char *) and initializes this->code;
		  *
		  */
		 inline explicit MatrixError (int code_ /**< Custom error code. */, const char* what_arg /**< Custom error message. */): runtime_error(what_arg), code(code_) {

		 }

		 /**
		  * @return this->code
		  */
		 inline int Code() { return code; }
	};

	/** This class implements a M * N Matrix.
	 *  It follows the rule of five as required by C++11
	 */
	class Matrix {
		protected:
			double * buffer = nullptr;
			unsigned rows = 0;
			unsigned columns = 0;

		private:
			/** Releases allocated buffer.
			 *
			 */

			void Destroy();

			/** Sets buffer = nullptr and rows, columns = 0.
			 *
			 */
			void Clear();

		public:
			/** Releases allocated buffer and sets rows and columns to zero and buffer to nullptr.
			 *	Creates an empty matrix.
			 */
			inline Matrix() {}

		    /** Copy constructor */
			Matrix (const Matrix& other /**< The source matrix */) ;

		    /** Move constructor */
			Matrix (Matrix&& other /**< The source matrix */) noexcept;

			/** Copy operator */
			Matrix& operator = (const Matrix& other /**< The source matrix */);

			/** Move assignment operator */
			Matrix& operator = (Matrix&& other /**< The source matrix */) noexcept;


			inline double * Buffer() { return buffer;}; // Be careful with direct buffer manipulation!!

			/** Creates a rows_ * columns_ matrix.
			*
			*/
			Matrix(std::istream & input);

			/** Creates a rows_ * columns_ matrix.
			 *
			 */
			inline Matrix(
				unsigned rows_, /**< Vertical dimension*/
				unsigned columns_  /**< Horizontal dimension*/
			):
					rows(rows_), columns(columns_){ buffer = (double *)calloc(rows * columns, sizeof(double));}

			/** Creates a rows_ * columns_ matrix.
			 *
			 */
			Matrix(
				double init, /**< Initial value*/
				unsigned rows_,  /**< Vertical dimension*/
				unsigned columns_ /**< Horizontal dimension*/
			);

			/** Returns true if this matrix is empty.
			 *
			 */
			inline bool IsEmpty() {return buffer == nullptr;}

			/** Returns a reference to the value @(row, column) position.
			 *  @return a double.
			 */
			inline double & operator () (unsigned row, unsigned column) {return buffer[row + column*rows];}

			/** Reports the vertical dimension.
			 * @return Row count.
			 */
			inline unsigned Rows() { return rows; }

			/** Reports the horizontal dimension.
			 * @returns count.
			 */
			inline unsigned Columns() {return columns;}

			/** Store the same value in all cells.
			 * @param v
			 */
			void All(double v);
			/** Releases allocated buffer and sets rows and columns to zero and buffer to nullptr.
			 *
			 */
			inline ~Matrix() {Destroy();}
	};
}



#endif /* MATRIX_H_ */
