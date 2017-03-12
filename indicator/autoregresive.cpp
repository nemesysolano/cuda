#include "autoregresive.h"

using namespace indicator;
using namespace timeseries;

/** Creates the internal buffer that represents the lookback period.
 *
 */
PartialAutocorrelation::PartialAutocorrelation(
	unsigned length, /**< Lookback period's length. */
	timeseries::TimeSeries & series /**< Bag containing the timeseries references that feed this indicator. */
): Indicator(length, TimeSeriesBag(series)){

}
