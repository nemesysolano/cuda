/*
 * object-pool.h
 *
 *  Created on: Mar 12, 2017
 *      Author: rsolano
 */

#ifndef OBJECT_POOL_H_
#define OBJECT_POOL_H_
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>

namespace pool {
	/** Error code ranging between 0 and 100 are reserved for exceptions thrown from IndicatorError's methods.
	 *
	 */
	enum POOL_ERROR_ENUM {
		INVALID_START_THREAD,
		INVALID_STATE_FOR_ADDING,
		INVALID_STATE_FOR_SUBMISSION,
		POOL_ALREADY_DESTROYED
	};

	extern const char* POOL_ERROR_MESSAGES[];

	/** This exception reports synchronization/threading issues inside Pool instances.
	 *
	 */
	class PoolError: public std::runtime_error {
	protected:
		int code = 0 /**< Custom error message. */;
	public:
		/** calls runtime_error(const string &) and initializes this->code;
		 *
		 */
		 inline explicit PoolError (int code_ /**< Custom error code. */, const std::string& what_arg /**< Custom error message. */): runtime_error(what_arg), code(code_) {

		 }

		 /** calls runtime_error(const char *) and initializes this->code;
		  *
		  */
		 inline explicit PoolError (int code_ /**< Custom error code. */, const char* what_arg /**< Custom error message. */): runtime_error(what_arg), code(code_) {

		 }

		 /**
		  * @return this->code
		  */
		 inline int Code() { return code; }
	};



	template<typename T> class Pool{
	private:

		std::list<std::shared_ptr<T>> * all = nullptr;
		std::list<std::shared_ptr<T>> * available = nullptr;
		std::mutex * critical_section_mutex = nullptr;
		std::mutex * life_cycle_mutex = nullptr;
		std::thread * scheduler_thread = nullptr;
	    std::condition_variable * condition_var = nullptr;
	    bool * running = false;

		std::thread::id current_thread_id;
		bool original = true;

	public:

		Pool(const Pool&);

		Pool(Pool&&) = delete;

		Pool & operator=(const Pool&);

		Pool & operator=(Pool&&) = delete;

		Pool();

		~Pool();

		void Add(std::shared_ptr<T> & ptr);

		void Add(std::shared_ptr<T> &&  ptr);

		void Start();

		void Stop();

		void Submit(std::function<int (const T &)>  callback) ;


	};


}

#endif /* OBJECT_POOL_H_ */
