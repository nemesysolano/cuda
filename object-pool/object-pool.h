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
	    bool * running =  nullptr;

		std::thread::id current_thread_id;
		bool original = true;

	public:
		Pool(Pool&&) = delete;

		Pool & operator=(Pool&&) = delete;

		Pool() {
			current_thread_id = std::this_thread::get_id();
			all = new std::list<std::shared_ptr<T>>();
			available = new std::list<std::shared_ptr<T>>();
			critical_section_mutex = new std::mutex();
			life_cycle_mutex = new std::mutex();
			scheduler_thread = new std::thread();
		    condition_var = new std::condition_variable();
		    running = new bool(false);
		}

		Pool(const Pool& other){
			operator = (other);
		}

		Pool<T> & operator=(const Pool& other){
			current_thread_id = other.current_thread_id;
			all = other.all;
			available =other.available;
			critical_section_mutex = other.critical_section_mutex;
			life_cycle_mutex = other.life_cycle_mutex;
			scheduler_thread = other.scheduler_thread;
		    condition_var = other.condition_var;
		    running = other.running;

		    original = false;

			return *this;
		}


		~Pool() {
			if(original) {
				if(all != nullptr){
					delete all;
					delete available;
					delete critical_section_mutex;
					delete life_cycle_mutex;
					delete scheduler_thread;
					delete condition_var;
					delete running;

					all = nullptr;
					available = nullptr;
					critical_section_mutex = nullptr;
					life_cycle_mutex = nullptr;
					scheduler_thread = nullptr;
					condition_var = nullptr;
					running = nullptr;
				}else
					throw PoolError(POOL_ALREADY_DESTROYED, POOL_ERROR_MESSAGES[POOL_ALREADY_DESTROYED]);
			}
		}

		void Add(std::shared_ptr<T> & ptr){
			bool & running = *this->running;
			if(current_thread_id != std::this_thread::get_id() || running)
				throw PoolError(INVALID_STATE_FOR_ADDING, POOL_ERROR_MESSAGES[INVALID_STATE_FOR_ADDING]);

			if(!running) {
				this->all->push_back(ptr);
				this->available->push_back(ptr);
			}
		}

		void Add(std::shared_ptr<T> &&  ptr){
			std::shared_ptr<T> & ref = ptr;
			Add(ref);
		}

		void Start() {
			bool & running = *this->running;
			if(!running) {
				std::unique_lock<std::mutex> lock(*life_cycle_mutex);
				running = true;
			}
			condition_var->notify_all();
		}

		void Stop() {
			bool & running = *this->running;
			if(running) {
				std::unique_lock<std::mutex> lock(*life_cycle_mutex);
				running =false;
			}
			condition_var->notify_all();
		}

		void Submit(std::function<int (const T &)>  callback) {
			bool & running = *this->running;
			if(!running)
				throw PoolError(INVALID_STATE_FOR_SUBMISSION, POOL_ERROR_MESSAGES[INVALID_STATE_FOR_SUBMISSION]);
			bool found = false;
			std::shared_ptr<T> ptr;

			while(!found) {
				{
					std::unique_lock<std::mutex> lock(*critical_section_mutex);
					if(this->available->size() > 0) {
						ptr =this->available->back();
						this->available->pop_back();
						found = true;
					}
				}

				if(!found){
					std::unique_lock<std::mutex> lock(*critical_section_mutex);
					condition_var->wait(lock);
				}


				if(!running)
					break;
			}

			if(found) {
				T * ref = ptr.get();
				callback(*ref); //TODO: Handle exceptions in a nicely.

				{
					std::unique_lock<std::mutex> lock(*critical_section_mutex);
					this->available->push_front (ptr);
				}

				condition_var->notify_all();
			}
		}



	};


}

#endif /* OBJECT_POOL_H_ */
