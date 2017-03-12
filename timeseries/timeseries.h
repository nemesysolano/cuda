/*
 * time-series.h
 *
 *  Created on: Mar 2, 2017
 *      Author: rsolano
 */

#ifndef TIME_SERIES_H_
#define TIME_SERIES_H_
#include <cmath>
#include <iostream>

using namespace std;
namespace timeseries {
	extern const double NOT_A_NUMBER;

	/** Abstract time series.
	 * Indicators and data feeds will be derived from this class.
	 * No parameter is validated in order to improve performance, therefore make sure you respect the documented constraints.
	 */
	class TimeSeries {
	protected:
		unsigned length = 0; /*!< the window length; defaults to 0*/
		double * window = nullptr; /*!< the data window; defaults to nullptr*/
		bool allocated=false; /*!< true if the window buffer was allocated by the constructor. */
		unsigned count = 0; /*!< */


	public:
		/** Initializes length property and optionally allocate this->window buffer.
		 *
		 */
		TimeSeries(
			unsigned length_ /**< [in] Strict positive integer. Indicates the loopback period'sYe length */,
			bool allocate /**< [in] When true, this constructor makes this->window = new double[length_]*/
		);

		/** Returns the head element. Namely window[length-(index+1)]
		 * @return A double
		 */
		double Head(
			unsigned index /**< [in]  Any integer between 0 and this.Count()-1*/
		) {
			return count > 0 ? window[length - (index+1)] : NOT_A_NUMBER;
		}

		/** Returns the tail element. Namely *(window + tail_index + index + 1)
		 * @return A double
		 */
		double Tail(
			unsigned index /**< [in]  Any integer between 0 and this.Count()-1*/
		) {

			return Head(count-(index+1));
		}

		/** Moves all items from right to left and makes this->window[this->length-1] = item
		 *	@return this instance
		 */
		TimeSeries & Push(double item /**< [in]  Any real value not equals to NaN. */);

		/** pushes 0 (this->Push(0)) into this time series
		 * @return 0
		 */
		virtual double Next();

		/**
		 * @return length - (tail_index + 1)
		 */
		inline unsigned Count() {
			return count;
		}

		/** this->length is lookback period's length.
		 * @return lookback period's length.
		 */
		inline unsigned Length() {
			return length;
		}

		/** Copies source.window to this->window.
		 *
		 */
		void copy(TimeSeries &source /**< [in]  Source reference. */);
#ifdef DEBUG
		void print();
#endif
		/** Destroys this->window if it was allocated in the constructor.
		 *
		 */
		virtual ~TimeSeries();
	};

}



#endif /* TIME_SERIES_H_ */
