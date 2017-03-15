#include <omp.h>
#include "matrix.h"
#include <stdlib.h>
#include <string.h>
#include <functional>
#include <algorithm>
#include <future>
#include <iterator>

using namespace matrix;
using namespace std;

const char* matrix::MATRIX_ERROR_MESSAGES[] = {
	"@Matrix::Matrix(stream & input): Unexpected EOF in matrix file.",
	"@Matrix::Matrix(stream & input): Can't read data from the provided input stream",
};
/** Creates a rows_ * columns_ matrix.
*
*/
Matrix::Matrix(
	istream & input
) {
	istream_iterator<double> iterator(input); /**< Wrapper std iterator that provides data to this instance. */
	istream_iterator<double> eos; /**< [in] End of stream flag */

	if(input.good()) {
		unsigned rows = (unsigned )*iterator;
		iterator++;
		unsigned columns = (unsigned )*iterator;
		unsigned check = rows * columns, count = 1;
		iterator ++;

		this->rows = rows;
		this->columns = columns;
		this-> buffer = (double *)malloc(rows * columns * sizeof(double));

		Matrix & self = *this;

		for(unsigned i = 0; i < rows; i++) {
			for(unsigned j = 0; j < columns; j++) {
				self(i,j) = * iterator;
				iterator++;
				count++;
				if(iterator == eos && count < check) {
					throw MatrixError(UNEXPECTED_EOF_IN_CONSTRUCTOR, MATRIX_ERROR_MESSAGES[UNEXPECTED_EOF_IN_CONSTRUCTOR]);
				}
			}
		}


	} else {
		throw MatrixError(BAD_INPUT_STREAM, MATRIX_ERROR_MESSAGES[BAD_INPUT_STREAM]);
	}


}

/** Creates a rows_ * columns_ matrix.
 *
 */
Matrix::Matrix(
	double init, /**< Initial value*/
	unsigned rows_,  /**< Vertical dimension*/
	unsigned columns_ /**< Horizontal dimension*/
):
	rows(rows_), columns(columns_)
{
	const unsigned length = rows * columns;
	buffer = (double *)malloc(rows * columns * sizeof(double));

#pragma omp parallel
#pragma omp for
    for(unsigned i = 0; i < length; i++)
    {
    	buffer[i] = init;
    }
}

/** Store the same value in all cells.
 * @param v
 */
void Matrix::All(double v) {
	const unsigned length = rows * columns;
	double * buffer = this->buffer;

#pragma omp parallel
#pragma omp for
    for(unsigned i = 0; i < length; i++)
    {
    	buffer[i] = v;
    }
}

/** Copy constructor */
Matrix::Matrix (const Matrix& other /**< The source matrix */) :
	rows(other.rows), columns(other.columns)
{
	buffer = (double *)malloc(rows * columns * sizeof(double));
	memcpy(buffer, other.buffer, rows*columns*sizeof(double));
}



/** Move constructor */
Matrix::Matrix (Matrix&& other /**< The source matrix */) noexcept:
		buffer(other.buffer), rows(other.rows), columns(other.columns) {
	other.Clear();
}

/** Copy assignment operator */
Matrix& Matrix::operator = (const Matrix& other /**< The source matrix */)
{
	Matrix tmp(other);         // re-use copy-constructor
    *this = std::move(tmp); // re-use move-assignment
    return *this;
}

/** Move assignment operator */
Matrix& Matrix::operator = (Matrix&& other /**< The source matrix */) noexcept
{
    Destroy();
    this->buffer = other.buffer;
    this->rows = other.rows;
    this->columns = other.columns;

    other.Clear();
    return *this;
}

void Matrix::Clear() {
	this->buffer = nullptr;
	this->rows = 0;
	this->columns = 0;
}

void Matrix::Destroy() {
	if(buffer != nullptr) {
		free(buffer);
		Clear();
	}
}
