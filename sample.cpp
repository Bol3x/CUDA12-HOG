#include <iostream>
#include <windows.h>

using namespace std;

int main(){
	int nBin_rows = 9;
	double ***HOGBin = new double**[nBin_rows];
	for(int i = 0; i < nBin_rows; ++i){
		HOGBin[i] = new double*[nBin_rows];
	}

	for(int i = 0; i < nBin_rows; ++i){
		for(int j = 0; j < nBin_rows; ++j){
			HOGBin[i][j] = new double[9] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		}
	}
	
	for (int i = 0; i < nBin_rows; ++i){
		for(int j = 0; j < nBin_rows; ++j){
			delete[] HOGBin[i][j];
		}
		delete[] HOGBin[i];
	}
	delete[] HOGBin;

}