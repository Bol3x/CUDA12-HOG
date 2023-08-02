#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_hog.cuh"

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


void get_HOG_features(double* HOG_features, Mat img, double*** HOGBin, int rows, int cols) {

	const double bins[10] = { 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0 };

	/***********************************************
	*					COMPUTE BINS
	************************************************/
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			double x = (i == 0 || i == rows - 1) ? 0 : img.at<unsigned char>(i + 1, j) - img.at<unsigned char>(i - 1, j);
			double y = (j == 0 || j == cols - 1) ? 0 : img.at<unsigned char>(i, j + 1) - img.at<unsigned char>(i, j - 1);

			double mag = sqrt(x * x + y * y);
			double dir = atan2(y, x) * 180 / M_PI;
			if (dir < 0) dir += 180;
			if (dir == 180) dir = 0;

			int HOG_row = i / 8;
			int HOG_col = j / 8;

			int bin_key = dir / 20.0;
			bin_key %= 9;

			double bin_value_lo = ((bins[bin_key + 1] - dir) / 20.0) * mag;
			double bin_value_hi = fabs(bin_value_lo - mag);

			HOGBin[HOG_row][HOG_col][bin_key] += bin_value_lo;
			HOGBin[HOG_row][HOG_col][(bin_key + 1) % 9] += bin_value_hi;
		}
	}

	/***********************************************
	*				NORMALIZE GRADIENTS
	************************************************/

	const int hog_rows = rows / 8;
	const int hog_cols = cols / 8;

	const int features = (hog_rows - 1) * (hog_cols - 1) * 36;

	int feature_index = 0;
	for (int i = 0; i < hog_rows - 1; i++) {
		for (int j = 0; j < hog_cols - 1; j++) {
			double sum = 0;
			
			#pragma unroll
			for (int k = 0; k < 9; k++) {
				double temp0, temp1, temp2, temp3;
				temp0 = HOGBin[i][j][k];
				temp1 = HOGBin[i][j + 1][k];
				temp2 = HOGBin[i + 1][j][k];
				temp3 = HOGBin[i + 1][j + 1][k];
				HOG_features[feature_index + k] = temp0;
				HOG_features[feature_index + 9 + k] = temp1;
				HOG_features[feature_index + 18 + k] = temp2;
				HOG_features[feature_index + 27 + k] = temp3;

				sum += (temp0 * temp0) + (temp1 * temp1) + (temp2 * temp2) + (temp3 * temp3);
			}

			#pragma unroll
			for (int k = 0; k < 36; k++) {
				HOG_features[feature_index + k] /= sqrt(sum + (1e-6 * 1e-6));
			}

			feature_index += 36;
		}
	}
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


	cout << "Total Average Time (in us): " << time_elapsed_bin + time_elapsed_norm << endl;


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
	double* C_HOG_features = (double*)malloc(sizeof(double) * norm_elems);

	get_HOG_features(C_HOG_features, image, HOGBin, rows, cols);

	long err_count = 0;
	for (int i = 0; i < block_rows; i++) {
		for (int j = 0; j < block_cols; j++) {
			for (int k = 0; k < 9; k++) {
				double cuda = HOG_features[(i*block_cols*9) + (j*9) + k];
				double c = HOGBin[i][j][k];
				if (abs(cuda - c) > 0.1) {
					err_count++;
					cout << (i * block_cols * 9) + (j * 9) + k << "\t" << cuda << "\t" << c << endl;
				}
			}
		}
	}
	cout << "Error Count: " << err_count << endl;

	err_count = 0;
	for (int i = 0; i < norm_elems; i++) {
		double cuda = normHOG[i];
		double c = C_HOG_features[i];

		if (abs(cuda - c) > 0.0001) {
			err_count++;
			cout << i << "\t" << cuda << "\t" << c << endl;
		}
	}

	cout << "Error Count: " << err_count << endl;

	free(C_HOG_features);
	cudaFree(norm_coeff);
	cudaFree(normHOG);
	cudaFree(HOG_features);

	return 0;
}