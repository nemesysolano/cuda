#include <omp.h>
#include "matrix.h"
#include <stdlib.h>
#include <string.h>
#include <functional>
#include <algorithm>
#include <future>

using namespace matrix;
using namespace std;

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
