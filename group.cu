#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_hog.cuh"
#include "hog_visualize.h"
#include "c_hog_features.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

/**
*	debug function to see pixel blocks
*/
void displayBlock(float *arr, int cols) {
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			cout << arr[(i * cols) + j] << "\t";
		}
		cout << endl;
	}
	cout << endl;
}



/**********************************************
*				MAIN PROGRAM
***********************************************/
int main() {

	//testing variables
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float time_elapsed = 0, time_elapsed_bin = 0, time_elapsed_norm = 0;
	float time_elapsed_c = 0;
	int runs = 100;

	/************************************************************
	*					1. Reading image data
	*************************************************************/

	string image_path = "C:\\Users\\Carlo\\Downloads\\images\\shiba_inu_69.jpg";

	//greyscale for now, we can update later
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
	short block_size = 8;

	//pad image to make it divisible by block_size
	resize(image, 
			image, 
			Size(image.cols - (image.cols % block_size), 
				 image.rows - (image.rows % block_size))
	);

	Size img_size = image.size();
	int rows = img_size.height;
	int cols = img_size.width;

	cout << "Image dimensions: " << endl;
	cout << image.rows << "x" << image.cols << "\n" << endl;
	imshow("Input", image);

	//transfer image data to unified memory
	unsigned char* img_data;
	cudaMallocManaged(&img_data, img_size.area() * sizeof(char));
	memcpy(img_data, image.ptr(), img_size.area() * sizeof(char));

	//initialize kernel parameters
	const int BLOCK_SIZE = 32;
	dim3 threadBlock(BLOCK_SIZE, BLOCK_SIZE);	//32*32 (1024 threads)
	dim3 dimBlock(rows / BLOCK_SIZE, cols / BLOCK_SIZE);

	/************************************************************
	*					2-4.  HOG Features
	*************************************************************/

	//initialize HOG Features as flattened 3d array
	//indexing: HOG_features[(i * cols * 9) + (j * 9) + k]
	int numFeatures = img_size.area() / 64 * 9;
	double* HOG_features;

	cudaMallocManaged(&HOG_features, numFeatures * sizeof(double));
	cudaMemset(HOG_features, 0, numFeatures * sizeof(double));

	//call aux function
	for (int i = 0; i < runs; i++) {
		cudaMemset(HOG_features, 0, numFeatures * sizeof(double));

		cudaEventRecord(start);
		compute_bins <<< dimBlock, threadBlock >>> (HOG_features, img_data, rows, cols);
		cudaEventRecord(end);

		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time_elapsed, start, end);
		time_elapsed_bin += time_elapsed;
	}

	time_elapsed_bin = time_elapsed_bin * 1e3 / runs;

	/************************************************************
	*					5.  Normalization
	*************************************************************/

	int block_rows = rows / 8;
	int block_cols = cols / 8;
	int norm_elems = (block_rows - 1) * (block_cols - 1) * 36;

	double* normHOG;
	cudaMallocManaged(&normHOG, norm_elems * sizeof(double));

	// Define the block and grid dimensions for the kernel
	const int numThreads = 1024;
	const int numBlocks = (norm_elems + numThreads - 1) / numThreads;

	double* norm_coeff;
	cudaMallocManaged(&norm_coeff, sizeof(double) * (norm_elems / 36));


		for (int i = 0; i < runs; i++) {

			cudaEventRecord(start);
			// Launch the kernel for L2 normalization on each 1x36 feature
			copyBinData <<< numBlocks, numThreads >>> (normHOG, HOG_features, block_rows, block_cols, norm_elems);
			L2norm <<< numBlocks, numThreads >>> (normHOG, norm_coeff, norm_elems);
			cudaEventRecord(end);

			cudaEventSynchronize(end);
			cudaEventElapsedTime(&time_elapsed, start, end);
			time_elapsed_norm += time_elapsed;
		}

	time_elapsed_norm = time_elapsed_norm * 1e3 / runs;


	/************************************************************
	*					6. HOG Visualization
	*************************************************************/

	cout << "Bin: Average time elapsed (in us): " << time_elapsed_bin << endl;
	cout << "Norm: Average time elapsed (in us): " << time_elapsed_norm << endl;

	double*** CUDA_HOG_bin = new double** [block_rows];
	for (int i = 0; i < block_rows; ++i) {
		CUDA_HOG_bin[i] = new double* [block_cols];
		for (int j = 0; j < block_cols; ++j) {
			CUDA_HOG_bin[i][j] = new double[9] 
			{
				HOG_features[(i * block_cols * 9) + (j * 9)],
				HOG_features[(i * block_cols * 9) + (j * 9) + 1],
				HOG_features[(i * block_cols * 9) + (j * 9) + 2],
				HOG_features[(i * block_cols * 9) + (j * 9) + 3],
				HOG_features[(i * block_cols * 9) + (j * 9) + 4],
				HOG_features[(i * block_cols * 9) + (j * 9) + 5],
				HOG_features[(i * block_cols * 9) + (j * 9) + 6],
				HOG_features[(i * block_cols * 9) + (j * 9) + 7],
				HOG_features[(i * block_cols * 9) + (j * 9) + 8],
			};
		}
	}

	//visualize HOG features of CUDA output
	visualizeHOG(image, CUDA_HOG_bin, block_rows, block_cols);

	cout << endl;

	cout << "Total Average Time [CUDA] (in us): " << time_elapsed_bin + time_elapsed_norm << endl;


	/************************************************************
	*					7. C Comparison
	*************************************************************/

	double*** HOGBin = new double** [block_rows];
	for (int i = 0; i < block_rows; ++i) {
		HOGBin[i] = new double* [block_cols];
		for (int j = 0; j < block_cols; ++j) {
			HOGBin[i][j] = new double[9] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		}
	}
	double* C_normHOG = (double*)malloc(sizeof(double) * norm_elems);


	for (int i = 0; i < runs; i++) {

		//reset HOGBin
		for (int n = 0; n < block_rows; n++) {
			for (int m = 0; m < block_cols; m++) {
				for (int l = 0; l < 9; l++) {
					HOGBin[n][m][l] = 0;
				}
			}
		}

		cudaEventRecord(start);
		get_HOG_features(C_normHOG, image, HOGBin, rows, cols);
		cudaEventRecord(end);

		cudaEventSynchronize(end);
		cudaEventElapsedTime(&time_elapsed, start, end);
		time_elapsed_c += time_elapsed;
	}
	time_elapsed_c = time_elapsed_c * 1e3 / runs;

	cout << "Total Average Time [C] (in us): " << time_elapsed_c << endl;


	/************************************************************
	*						8. Error Check
	*************************************************************/

	//visualize HOG features of C output
	visualizeHOG(image, HOGBin, block_rows, block_cols);

	long err_count = 0;
	for (int i = 0; i < norm_elems; i++) {
		double cuda = normHOG[i];
		double c = C_normHOG[i];

		if (abs(cuda - c) > 0.0001) {
			err_count++;
			cout << i << "\t" << cuda << "\t" << c << endl;
		}
	}

	cout << "Feature Error Count: " << err_count << endl;

	/************************************************************
	*						Free Memory
	*************************************************************/

	// Free memory
	for (int i = 0; i < block_rows; ++i) {
		for (int j = 0; j < block_cols; ++j) {
			delete[] HOGBin[i][j];
			delete CUDA_HOG_bin[i][j];
		}
		delete[] HOGBin[i];
		delete[] CUDA_HOG_bin[i];
	}
	delete[] HOGBin;
	delete[] CUDA_HOG_bin;

	free(C_normHOG);
	cudaFree(img_data);
	cudaFree(norm_coeff);
	cudaFree(normHOG);
	cudaFree(HOG_features);

	return 0;
}