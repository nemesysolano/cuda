/*
 * autoregresive.h
 *
 *  Created on: Mar 6, 2017
 *      Author: rsolano
 */

#ifndef AUTOREGRESIVE_H_
#define AUTOREGRESIVE_H_
#include "indicator.h"

namespace indicator {
	class PartialAutocorrelation: public Indicator {
		public:
		/** Creates the internal buffer that represents the lookback period.
		 *
		 */
		PartialAutocorrelation(
			unsigned length, /**< Lookback period's length. */
			timeseries::TimeSeries & series /**< Bag containing the timeseries references that feed this indicator. */
		);

	};
}


#endif /* AUTOREGRESIVE_H_ */
