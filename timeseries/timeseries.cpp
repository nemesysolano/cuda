#include "timeseries.h"
#include <cstdlib>
#include <iostream>
#include <limits>
#include <cmath>
#include <cstring>

using namespace std;
using namespace timeseries;

const double timeseries::NOT_A_NUMBER = std::numeric_limits<double>::quiet_NaN();

/** Initializes length property
 *	No parameter is validated in order to improve performance, therefore
 *	make sure you respect the documented constraints.
 */
TimeSeries::TimeSeries(
	unsigned length_ /**< [in] Strict positive integer. Indicates the window length */,
	bool allocate /**< [in] When true, this constructor makes this.window = new double[length_]*/
):length(length_){
	if(allocate) {

#ifdef DEBUG
		cout << "DEBUG: allocating " << this->length << " elements" << endl;
#endif
		this->allocated = true;
		this->window = (double *)malloc(sizeof(double) * this->length);

	}
}

/** Copy constructor */
TimeSeries::TimeSeries (const TimeSeries& other /**< The source matrix */): TimeSeries(other.length, true){
	this->count = other.count;
	memcpy(this->window, other.window, length * sizeof(double));
}

/** Move constructor */
TimeSeries::TimeSeries (TimeSeries&& other /**< The source matrix */) noexcept {
	Move(other);
}

/** Copy operator */
TimeSeries& TimeSeries::operator = (const TimeSeries& other /**< The source matrix */){
	this->Destroy();
	this->length = other.length; /*!< the window length; defaults to 0*/
	this->window = (double *)malloc(sizeof(double) * other.length); /*!< the data window; defaults to nullptr*/
	this->allocated= true; /*!< true if the window buffer was allocated by the constructor. */
	this->count = other.count;

	memcpy(this->window, other.window, length * sizeof(double));

	return * this;
}

/** Move assignment operator */
TimeSeries& TimeSeries::operator = (TimeSeries&& other /**< The source matrix */) noexcept{
	Move(other);
	return * this;
}

/**
 * Copies (direct assigment) other's properties into this and then clears all other's properties.
 */
void TimeSeries::Move(
	TimeSeries & other /*!< The source time series. */
) {
	this->length = other.length; /*!< the window length; defaults to 0*/
	this->window = other.window; /*!< the data window; defaults to nullptr*/
	this->allocated= other.allocated; /*!< true if the window buffer was allocated by the constructor. */
	this->count = other.count;
	other.Clear();
}

/** Sets all all integer / pointer properties to zero / nullptr.
 *
 */
void TimeSeries::Clear() {
	this->length = 0; /*!< the window length; defaults to 0*/
	this->window = nullptr; /*!< the data window; defaults to nullptr*/
	this->allocated=false; /*!< true if the window buffer was allocated by the constructor. */
	this->count = 0; /*!< */
}

/** Releases all bufers and set all integer / pointer variables to zero / nullptr.
 *
 */
void TimeSeries::Destroy(){
	if(allocated) {
#ifdef DEBUG
		cout << "DEBUG: deleting " << this->length << " elements" << endl;
#endif
		delete this->window;
	}

	Clear();
}


/** Moves all items from right to left and makes this->window[this->length-1] = item
 *	@return this instance
 */
TimeSeries & TimeSeries::Push(double item /**< [in]  Any real value not equals to NaN. */) {
	register double * window = this->window;

	if(window != nullptr) {
		register unsigned i;
		unsigned lower = this->length - count;

		for(i = 0; i < length-1; i++) {
			window[i] = window[i+1];
		}

		window[this->length -1] = item;

		if(count < length){
			count++;
		}
	}

	return *this;
}

/** pushes 0 (this->Push(0)) into this time series
 * @return 0
 */
double TimeSeries::Next(){
	Push(0);
	return 0;
}

#ifdef DEBUG
void TimeSeries::print() {
	unsigned i;
	for(i = 0; i < length - 1; i++) {
		cout << window[i] << ',';
	}

	cout << window[i] << endl;
}
#endif

/** Copies source.window to this->window.
 *
 */
void TimeSeries::copy(TimeSeries &source /**< [in]  Source reference. */) {
	unsigned source_length = source.Length();
	unsigned actual_length = source_length > length ? length: source_length;

}
/** Destroys this->window if it was allocated in the constructor.
 *
 */
TimeSeries::~TimeSeries() {
	Destroy();
}
