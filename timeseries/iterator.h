/*
 * iterator.h
 *
 *  Created on: Mar 3, 2017
 *      Author: rsolano
 */

#ifndef ITERATOR_H_
#define ITERATOR_H_
#include "timeseries.h"
#include <iostream>
#include <iterator>

namespace timeseries{
	/**
	 * This iterator implements a serial/forward data feed for IteratorTimeSeries instances.
	 */
	class Iterator  {
	public:

		/**
		 * @returns averages::NOT_A_NUMBER when iterator can't fetch more values.
		 */
		virtual double operator*() = 0;

		/** Postfix increment operator
		 * Do not call this operator when this instance is referenced by StreamIterator.
		 * @returns *this
		 */
		virtual Iterator & operator++(int) = 0;

		/** Added to avoid warnings.
		 */
		virtual ~Iterator();

	};
}


#endif /* ITERATOR_H_ */
