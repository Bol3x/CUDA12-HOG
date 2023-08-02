#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_hog.cuh"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <time.h>

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
	clock_t start, end;
	double time_elapsed_grad = 0, time_elapsed_bin = 0, time_elapsed_norm = 0;
	int runs = 100;

	/************************************************************
	*					1. Reading image data
	*************************************************************/

	string image_path = "C:\\Users\\Carlo\\Downloads\\images\\shiba_inu_60.jpg";

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

		start = clock();
		compute_bins <<< dimBlock, threadBlock >>> (HOG_features, img_data, rows, cols);
		cudaDeviceSynchronize();
		end = clock();

		time_elapsed_bin += ((double)(end - start));
	}

	time_elapsed_bin = time_elapsed_bin * 1e6 / CLOCKS_PER_SEC / runs;

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

			start = clock();

			// Launch the kernel for L2 normalization on each 1x36 feature
			L2norm <<< numBlocks, numThreads >>> (normHOG, HOG_features, norm_coeff, block_rows, block_cols, norm_elems);
			cudaDeviceSynchronize();

			end = clock();

			time_elapsed_norm += ((double)(end - start));
		}

	time_elapsed_norm = time_elapsed_norm * 1e6 / CLOCKS_PER_SEC / runs;


	/************************************************************
	*					6. HOG Visualization
	*************************************************************/

	cout << "Bin: Average time elapsed (in us): " << time_elapsed_bin << endl;
	cout << "Norm: Average time elapsed (in us): " << time_elapsed_norm << endl;

	cout << "HOG norm" << endl;
	for (int i = 36; i < 36 + 36; i++) {
		cout << normHOG[i] << "\t";
	}
	cout << endl;

	cudaFree(norm_coeff);
	cudaFree(normHOG);
	cudaFree(HOG_features);

	return 0;
}