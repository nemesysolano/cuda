#include "object-pool.h"

const char* pool::POOL_ERROR_MESSAGES[] = {
	"@Pool<T>::Scheduler(Pool & pool). columns_ > rows_",
	"@Pool<T>::Add(std::shared_ptr<T> & ptr). The calling thread is not the same one that created this instance or the scheduler is running."
	"@Pool<T>::Submit(Callback callback). The scheduler thread is stopped."
};


using namespace pool;

