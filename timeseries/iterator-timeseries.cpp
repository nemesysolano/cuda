#include "iterator-timeseries.h"

using namespace timeseries;
using namespace std;

/** Fetches the next value from the internal iterator.
 * If the fetched value is a number, it's pushed into the sequence's, otherwise sequence's state remains unchanged.
 * @return The value just fetched from the iterator.
 */
double IteratorTimeSeries::Next() {
	double current = *iterator;
	if(isnan(current))
		return current;

	Push(current);
	iterator++;
	return current;
}

IteratorTimeSeries::~IteratorTimeSeries() {

}
