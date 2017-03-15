#include "stream-iterator.h"

using namespace timeseries;


/** Initializes internal istream_iterator
 *
 */
DoubleIterator::DoubleIterator(
	std::istream & input_
) : iterator(std::istream_iterator<double>(input_)), input(input_) {
	(*this)++;
}
/**
 * @returns averages::NOT_A_NUMBER when iterator can't fetch more values.
 */
double DoubleIterator::operator*() {
	return current;
}

/** Postfix increment operator
 * @returns *this
 */
Iterator & DoubleIterator::operator++(int){
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
DoubleIterator::~DoubleIterator(){

}
