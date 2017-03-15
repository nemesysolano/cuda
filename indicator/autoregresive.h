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
#include <matrix.h>

namespace indicator {
	class PhiSampler {
	public:
		virtual matrix::Matrix & Fill (
			indicator::TimeSeriesBag & bag_,
			timeseries::TimeSeries & current,
			matrix::Matrix & phi_sample_matrix
		) = 0;
		virtual ~PhiSampler();
	};

	class PartialAutocorrelation: public Indicator {
		protected:
			matrix::Matrix phi_sample_matrix;
			PhiSampler & phi_sampler;
		public:

		/** Creates the internal buffer that represents the lookback period.
		 *
		 */
		PartialAutocorrelation(
			unsigned length, /**< Lookback period's length. */
			indicator::TimeSeriesBag & bag_, /**< Bag containing the timeseries references that feed this indicator. */
			pool::Pool<regression::LinearSolver> & solver_pool_, /**< This pool provides LinearSolver instances used to perform multivariate regression. */
			PhiSampler & phi_sampler_ /**< Generates samples of phi coefficients. */
		);

		inline const PhiSampler & Sampler() { return phi_sampler; }

		virtual double Next();
	};


	class VolumeDeltaSampler: public PhiSampler {
		virtual matrix::Matrix & Fill (
			indicator::TimeSeriesBag & bag_, /**< bag_[0] is a price series and bag_[1] is a volume series */
			timeseries::TimeSeries & current,
			matrix::Matrix & phi_sample_matrix
		);
	};
}


#endif /* AUTOREGRESIVE_H_ */
