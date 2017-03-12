#include "stream-iterator.h"

using namespace timeseries;


/** Initializes internal istream_iterator
 *
 */
StreamIterator::StreamIterator(
	std::istream & input_
) : iterator(std::istream_iterator<double>(input_)), input(input_) {
	(*this)++;
}
/**
 * @returns averages::NOT_A_NUMBER when iterator can't fetch more values.
 */
double StreamIterator::operator*() {
	return current;
}

/** Postfix increment operator
 * @returns *this
 */
Iterator & StreamIterator::operator++(int){
	if(iterator == eos) {
		current = timeseries::NOT_A_NUMBER;
	}else {
		current = *iterator;
		iterator ++;
	}

	return *this;
}

/**
 *
 */
StreamIterator::~StreamIterator(){

}
