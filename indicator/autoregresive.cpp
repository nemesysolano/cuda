#include "autoregresive.h"

using namespace indicator;
using namespace timeseries;
using namespace matrix;
using namespace regression;

/** Creates the internal buffer that represents the lookback period.
 *
 */
PartialAutocorrelation::PartialAutocorrelation(
	unsigned length, /**< Lookback period's length. */
	indicator::TimeSeriesBag & bag_, /**< Bag containing the timeseries references that feed this indicator. */
	pool::Pool<regression::LinearSolver> & solver_pool_, /**< This pool provides LinearSolver instances used to perform multivariate regression. */
	PhiSampler & phi_sampler_ /**< Generates samples of phi coefficients. */
):phi_sample_matrix(Matrix(length,length)), phi_sampler(phi_sampler_),Indicator(length, bag_, solver_pool_) {

}

double PartialAutocorrelation::Next(){
	phi_sampler.Fill(bag, *this, phi_sample_matrix);
	this->solver_pool.Submit(
		[] (const LinearSolver & solver) -> int {
			return 0;
		}
	);

	return 0;
}

matrix::Matrix &VolumeDeltaSampler::Fill (
		indicator::TimeSeriesBag & bag_, /**< bag_[0] is a price series and bag_[1] is a volume series */
		timeseries::TimeSeries & current,
		matrix::Matrix & phi_sample_matrix
	) {
	TimeSeries price = *bag_[0].get();
	TimeSeries volume = *bag_[1].get();
	const unsigned rows = current.Count();
	const unsigned columns = current.Count();

	phi_sample_matrix.All(0);
#pragma omp parallel for schedule(dynamic,1)

	for (unsigned i = 0; i < rows - 1; i++) {
		// only one i per thread
		for (unsigned j = 0; j < i+1; j++) {

		}
	}
	return phi_sample_matrix;
}
