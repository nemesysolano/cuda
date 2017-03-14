/*
 * autoregresive.h
 *
 *  Created on: Mar 6, 2017
 *      Author: rsolano
 */

#ifndef AUTOREGRESIVE_H_
#define AUTOREGRESIVE_H_
#include "indicator.h"
#include <object-pool.h>
#include <linear-solver.h>

namespace indicator {
	class PartialAutocorrelation: public Indicator {
		private:
			pool::Pool<regression::LinearSolver> & solvers_pool;

		public:
		/** Creates the internal buffer that represents the lookback period.
		 *
		 */
		PartialAutocorrelation(
			unsigned length, /**< Lookback period's length. */
			timeseries::TimeSeries & series /**< Bag containing the timeseries references that feed this indicator. */,
			pool::Pool<regression::LinearSolver> & solvers_pool /**< Object pool that provides linear solvers to this indicator. */
		);

	};
}


#endif /* AUTOREGRESIVE_H_ */
