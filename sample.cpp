#include <iostream>
#include <windows.h>

using namespace std;

int main(){
	int rows, cols;

	cin >> rows;
	cin >> cols;

	double **arr;
	arr = new double *[rows];

	for (int i = 0; i < rows; ++i){
		arr[i] = new double[cols];
	}

	for (int i = 0; i < rows; ++i){
		delete[] arr[i];
	}
	delete[] arr;



    return 0;
}