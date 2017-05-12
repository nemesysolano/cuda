/*
 ============================================================================
 Name        : regression-test.cu
 Author      : Rafael Solano
 Version     :
 Copyright   : (c) 2017
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <contrib/utilities.h>
#include <linear-solver.h>
#include <matrix.h>

using namespace matrix;
using namespace regression;
using namespace std;


void test_improved_code(const unsigned rows, const unsigned columns, LinearSolver & solver, const char * label) {
	Matrix A(rows, columns), B(rows, rows);
    /* */
    for(int row = 0; row < rows; row++){
        for(int column = 0; column < columns; column++) {
            A(row,column) = (column + row*row) * sqrt((double)(column + row));
        }
    }
    /* */

    try{
    	cout << label << ' ' << "start" << endl;
		Matrix R = solver(A, B);


		/* */
		for(int row = 0; row < R.Rows(); row++){
			for(int column = 0; column < R.Columns(); column++) {
				printf("%8.3f,",R(row,column));
			}
			printf("\n");
		}
    }catch(LinearSolverError &e) {
    	cout << e.what() << endl;
    }
    cout << label << ' ' << "ends" << endl;
    /* */
}
void test_improved_code(const unsigned rows, const unsigned columns, LinearSolver && solver, const char * label) {
	LinearSolver & ref(solver);
	test_improved_code(rows, columns, ref, label);
}
void test_allocated_solver() {
    const unsigned rows = 7;
    const unsigned columns = 5;
    LinearSolver solver(rows, columns);
    test_improved_code(rows, columns, solver, "TEST: allocator constructor");

}

void test_copied_solver() {
    const unsigned rows = 7;
    const unsigned columns = 5;
    test_improved_code(rows, columns, LinearSolver(rows, columns), "TEST: copy constructor");
}

void test_op_copied_solver() {
    const unsigned rows = 7;
    const unsigned columns = 5;
    LinearSolver src(rows, columns);
    LinearSolver dst = std::move(src);

    test_improved_code(rows, columns, dst, "TEST: copy assignment");
}

void test_larger_dimension_solver() {
    const unsigned rows = 7+2;
    const unsigned columns = 5+2;
    LinearSolver src(rows, columns);

    test_improved_code(rows-2, columns-2, src, "TEST: larger dimension");
}

int main(int argc, char * argv[]) {
//	test_contributed_code(argc, argv);
	test_allocated_solver();
	test_copied_solver();
	test_op_copied_solver();
	test_larger_dimension_solver();
}
