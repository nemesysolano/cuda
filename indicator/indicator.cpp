#include "indicator.h"

using namespace indicator;
using namespace timeseries;

static const char* INDICATOR_ERROR_MESSAGES[] = {
	"Indicator::Indicator. Loopback period's length for every timeseries in the bag must be equals to constructo's length parameter."
};
/** Creates the internal buffer that represents the lookback period.
 *
 */
Indicator::Indicator(
		unsigned length, /**< Lookback period's length. */
		indicator::TimeSeriesBag & bag_, /**< Bag containing the timeseries references that feed this indicator. */
		pool::Pool<regression::LinearSolver> & solver_pool_
):TimeSeries(length, true), bag(bag_), solver_pool(solver_pool_){

	TimeSeriesBag::iterator index = bag.begin(), end = bag.end();

	while(index != end) {
		TimeSeries * timeseries = (*index).get();
		if(timeseries->Length() != length) {
			throw IndicatorError((int)LOOPBACK_PERIOD_LENGTH_MISMATCH, INDICATOR_ERROR_MESSAGES[(int)LOOPBACK_PERIOD_LENGTH_MISMATCH]);
		}
		index++;
	}
}
