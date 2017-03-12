/*
 * timeseries-bag.h
 *
 *  Created on: Mar 6, 2017
 *      Author: rsolano
 */

#ifndef TIMESERIES_BAG_H_
#define TIMESERIES_BAG_H_
#include <functional>
#include <vector>
#include <timeseries.h>
namespace indicator {
	/** This class is a container of time series references.
	 *	Developer must take care of scoping issues when storing references into collections.
	 */
	class TimeSeriesBag:  public std::vector<std::reference_wrapper<timeseries::TimeSeries>> {
	public:
		inline TimeSeriesBag() : std::vector<std::reference_wrapper<timeseries::TimeSeries>> () {

		}

		inline TimeSeriesBag(timeseries::TimeSeries &series): TimeSeriesBag()  {
			this->push_back(series);
		}

	};
}

#endif /* TIMESERIES_BAG_H_ */
