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
#include <memory>
namespace indicator {
	/** This class is a container of time series references.
	 *	Developer must take care of scoping issues when storing references into collections.
	 */
	class TimeSeriesBag:  public std::vector<std::shared_ptr<timeseries::TimeSeries>> {
	public:
		inline TimeSeriesBag() : std::vector<std::shared_ptr<timeseries::TimeSeries>> () {

		}

		inline TimeSeriesBag(std::shared_ptr<timeseries::TimeSeries> ptr): TimeSeriesBag()  {
			this->push_back(ptr);
		}

	};
}

#endif /* TIMESERIES_BAG_H_ */
