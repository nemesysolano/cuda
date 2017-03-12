/*
 * indicator.h
 *
 *  Created on: Mar 6, 2017
 *      Author: rsolano
 */

#ifndef INDICATOR_H_
#define INDICATOR_H_
#include <timeseries.h>
#include "timeseries-bag.h"
namespace indicator {

	/** Error code ranging between 0 and 100 are reserved for exceptions thrown from IndicatorError's methods.
	 *
	 */
	enum INDICATOR_ERROR_ENUM {
		LOOPBACK_PERIOD_LENGTH_MISMATCH = 0
	};

	/** This exception class reports constraint violations detected in data provided to indicators.
	 *
	 */
	class IndicatorError: public std::runtime_error {
	protected:
		int code = 0 /**< Custom error message. */;
	public:
		/** calls runtime_error(const string &) and initializes this->code;
		 *
		 */
		 inline explicit IndicatorError (int code_ /**< Custom error code. */, const string& what_arg /**< Custom error message. */): runtime_error(what_arg), code(code_) {

		 }

		 /** calls runtime_error(const char *) and initializes this->code;
		  *
		  */
		 inline explicit IndicatorError (int code_ /**< Custom error code. */, const char* what_arg /**< Custom error message. */): runtime_error(what_arg), code(code_) {

		 }

		 /**
		  * @return this->code
		  */
		 inline int Code() { return code; }
	};

	/** In the context of technical analysis, an indicator is a mathematical calculation based on a security's price and/or volume. The result is used to predict future prices.
	 */
	class Indicator: public timeseries::TimeSeries {
	protected:
		indicator::TimeSeriesBag & bag; /**< Bag containing the timeseries references that feed this indicator. */
	public:

		/** Creates the internal buffer that represents the lookback period.
		 *
		 */
		Indicator(
			unsigned length, /**< Lookback period's length. */
			indicator::TimeSeriesBag & bag_ /**< Bag containing the timeseries references that feed this indicator. */
		);

		/** Creates the internal buffer that represents the lookback period.
		 *
		 */
		inline Indicator(
			unsigned length, /**< Lookback period's length. */
			indicator::TimeSeriesBag && bag_ /**< Bag containing the timeseries references that feed this indicator. */
		): Indicator(length, bag_){}

	};

}


#endif /* INDICATOR_H_ */
