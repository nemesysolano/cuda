/*
 * streamed-timeseries.h
 *
 *  Created on: Mar 3, 2017
 *      Author: rsolano
 */

#ifndef STREAMED_TIMESERIES_H_
#define STREAMED_TIMESERIES_H_
#include "iterator.h"

namespace timeseries {
	class IteratorTimeSeries: public TimeSeries {
	protected:
		Iterator & iterator;
	public:

		/** Creates the internal buffer that represents the lookback period and catches the reference to the iterator instance.
		 *
		 */
		inline IteratorTimeSeries(
			unsigned length,
			Iterator & iterator_
		): TimeSeries(length, true), iterator(iterator_){

		}

		/** Fetches the next value from the internal iterator.
		 * @return The value just fetched from the iterator.
		 */
		virtual double Next();

		virtual ~IteratorTimeSeries();
	};
}




#endif /* STREAMED_TIMESERIES_H_ */
