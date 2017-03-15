#include "object-pool.h"

const char* pool::POOL_ERROR_MESSAGES[] = {
	"@Pool<T>::Scheduler(Pool & pool). columns_ > rows_",
	"@Pool<T>::Add(std::shared_ptr<T> & ptr). The calling thread is not the same one that created this instance or the scheduler is running."
	"@Pool<T>::Submit(Callback callback). The scheduler thread is stopped.",
	"@Pool<T>::~Pool(). This pool was destroyed by another thread.",
};


using namespace pool;

template<typename T> Pool<T>::Pool() {
	current_thread_id = std::this_thread::get_id();
	all = new std::list<std::shared_ptr<T>>();
	available = new std::list<std::shared_ptr<T>>();
	critical_section_mutex = new std::mutex();
	life_cycle_mutex = new std::mutex();
	scheduler_thread = new std::thread();
    condition_var = new std::condition_variable();
    running = new bool();
    (*running) = false;
}

template<typename T> Pool<T>::Pool(const Pool& other){
	operator = (other);
}

template<typename T> Pool<T> & Pool<T>::operator=(const Pool& other){
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


template<typename T> Pool<T>::~Pool() {
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

template<typename T> void Pool<T>::Add(std::shared_ptr<T> & ptr){
	bool running = *(this->running);
	if(current_thread_id != std::this_thread::get_id() || running)
		throw PoolError(INVALID_STATE_FOR_ADDING, POOL_ERROR_MESSAGES[INVALID_STATE_FOR_ADDING]);

	if(!running) {
		this->all.push_back(ptr);
		this->available.push_back(ptr);
	}
}

template<typename T> void Pool<T>::Add(std::shared_ptr<T> &&  ptr){
	std::shared_ptr<T> & ref = ptr;
	Add(ref);
}

template<typename T> void Pool<T>::Start() {
	bool running = *(this->running);
	if(!running) {
		std::unique_lock<std::mutex> lock(life_cycle_mutex);
		*(this->running) = true;
	}
	condition_var->notify_all();
}

template<typename T> void Pool<T>::Stop() {
	bool running = *(this->running);
	if(running) {
		std::unique_lock<std::mutex> lock(life_cycle_mutex);
		*(this->running) =false;
	}
	condition_var->notify_all();
}

template<typename T> void Pool<T>::Submit(std::function<int (const T &)>  callback) {
	bool running = *(this->running);
	if(!running)
		throw PoolError(INVALID_STATE_FOR_SUBMISSION, POOL_ERROR_MESSAGES[INVALID_STATE_FOR_SUBMISSION]);
	bool found = false;
	std::shared_ptr<T> ptr;

	while(!found) {
		{
			std::unique_lock<std::mutex> lock(critical_section_mutex);
			if(this->available.size() > 0) {
				ptr =this->available->back();
				this->available->pop_back();
				found = true;
			}
		}

		if(!found){
			std::unique_lock<std::mutex> lock(critical_section_mutex);
			condition_var->wait(lock);
		}


		if(!running)
			break;
	}

	if(found) {
		T * ref = ptr.get();
		callback(*ref); //TODO: Handle exceptions in a nicely.

		{
			std::unique_lock<std::mutex> lock(critical_section_mutex);
			this->available->push_front (ptr);
		}

		condition_var->notify_all();
	}
}



