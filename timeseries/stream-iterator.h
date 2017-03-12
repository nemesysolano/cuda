/*
 * stream-iterator.h
 *
 *  Created on: Mar 3, 2017
 *      Author: rsolano
 */

#ifndef STREAM_ITERATOR_H_
#define STREAM_ITERATOR_H_

#include "iterator.h"
namespace timeseries {
	/**This iterator reads double precision values from a text file.
	 * The file must contain one value per line. Finally, the mutation methods must not be called
	 * if some time series holds a reference to this object.
	 */
	class StreamIterator : public Iterator {
	protected:
		std::istream_iterator<double> iterator; /**< Wrapper std iterator that provides data to this instance. */
		std::istream_iterator<double> eos; /**< [in] End of stream flag */
		std::istream & input; /**< [in] A valid refernce to an input stream. */
		double current = timeseries::NOT_A_NUMBER; /**< The last value fetched by the iterator */

	public:
		StreamIterator(StreamIterator && source) = delete;
		StreamIterator(StreamIterator & source)  = delete;

		/** Initializes internal istream_iterator
		 *
		 */
		StreamIterator(
			std::istream & input_ /**< [in] A valid refernce to an input stream. */
		);

		/**
		 * @returns averages::NOT_A_NUMBER when iterator can't fetch more values.
		 */
		virtual double operator*();

		/** Postfix increment operator
		 * @returns *this
		 */
		virtual Iterator & operator++(int);

		/**
		 *
		 */
		virtual ~StreamIterator();
	};
}



#endif /* STREAM_ITERATOR_H_ */
