#include "timeseries.h"
#include <cstdlib>
#include <iostream>
#include <limits>
#include <cmath>

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
		this->window = new double[this->length];

	}
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
	if(allocated) {
#ifdef DEBUG
		cout << "DEBUG: deleting " << this->length << " elements" << endl;
#endif
		delete this->window;
		allocated = false;
	}
}
