//============================================================================
// Name        : object-pool-test.cpp
// Author      : Rafael Solano
// Version     :
// Copyright   : (c) 2017
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <object-pool.h>
#include <string>

using namespace pool;
using namespace std;

class POINT {
public:
	int x;
	int y;
} ;

int connect (const POINT & object) {
	cout << object.x << ',' << object.y << endl;
	return 0;
}

int main(int argc, char * argv[]) {
	Pool<POINT> client_pool;
	client_pool.Add(shared_ptr<POINT>(new POINT{0,0}));
	client_pool.Add(shared_ptr<POINT>(new POINT{0,1}));
	client_pool.Add(shared_ptr<POINT>(new POINT{1,0}));
	client_pool.Add(shared_ptr<POINT>(new POINT{1,1}));

	client_pool.Start();
	for(int i = 0; i < 10; i++) {
		client_pool.Submit(connect);
	}

	client_pool.Stop();
}
