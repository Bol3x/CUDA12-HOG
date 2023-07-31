#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <cassert>

using namespace cv;
using namespace std;

void displayBlock(Mat img) {
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			cout << img.at<float>(i, j) << "\t";
		}
		cout << endl;
	}
	cout << endl;
}

/*
* Computes the gradient magnitude and direction of each pixel in the image
*/
__global__
void computegrad_device(float *mag, float *dir, unsigned char* input, int rows, int cols) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	const int stride_i = blockDim.x * gridDim.x;
	const int stride_j = blockDim.y * gridDim.y;

	float x, y;
	float temp;
	//compute x gradients
	for (i; i < rows; i += stride_i) {
		for (j; j < cols; j += stride_j) {
			//do not include first and last cols (borders)
			y = (j == 0 || j == cols - 1) ?
				0 : input[(i * cols) + j+1] - input[(i * cols) + j-1];

			//do not include first and last rows (borders)
			x = (i == 0 || i == rows - 1) ?
				0 : input[((i + 1) * cols) + j] - input[((i - 1) * cols) + j];

			mag[(i * cols) + j] = sqrt(x * x + y * y);
			temp = atan2(y, x) * 180 / M_PI;
			if (temp < 0) temp += 180;
			dir[(i * cols) + j] = temp;
		}
	}
}

/*
*	wrapper function to compute gradients using CUDA
*	todo: delete prints in final version
*/
void compute_gradients(Mat& mag, Mat& dir, Mat& input_mat) {

	//number of bytes for the matrix - step is bytes per row
	const int BYTE_SIZE =  input_mat.rows * input_mat.cols * sizeof(float);
	const int ARRAY_SIZE = input_mat.rows * input_mat.cols;

	//create pointers to put into the kernel
	float* mag_out, * dir_out;
	unsigned char *input;

	//allocate unified memory to pointers
	cudaMallocManaged(&input, ARRAY_SIZE);
	cudaMallocManaged(&mag_out, BYTE_SIZE);
	cudaMallocManaged(&dir_out, BYTE_SIZE);

	//cout << "Copying data to input...\n" << endl;

	//copy matrix data to unified memory
	memcpy(input, input_mat.ptr(), BYTE_SIZE);

	//initialize kernel data
	const int BLOCK_SIZE = 32;
	dim3 threadBlock(BLOCK_SIZE, BLOCK_SIZE);	//32*32 (1024 threads)
	dim3 dimBlock(input_mat.rows / BLOCK_SIZE, input_mat.cols / BLOCK_SIZE);	//

	//cout << "Launching kernel...\n" << endl;

	//launch kernel
	computegrad_device <<< dimBlock, threadBlock >>> (mag_out, dir_out, input, input_mat.rows, input_mat.cols);
	cudaDeviceSynchronize();

	//cout << "Copying mag output into matrices...\n" << endl;

	//copy unified memory into gradient matrices
	memcpy(mag.ptr(), mag_out, BYTE_SIZE);

	//cout << "Copying mag output into matrices...\n" << endl;

	//copy unified memory into gradient matrices
	memcpy(dir.ptr(), dir_out, BYTE_SIZE);

	//cout << "Freeing memory...\n" << endl;

	//free memory allocated
	cudaFree(mag_out);
	cudaFree(dir_out);
	cudaFree(input);
}

__global__
void group_bin(int n, double* hog_out, float* mag_in, float* dir_in, int rows, int cols) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;

	int row_idx;
	int col_idx;

	int bin_key;
	float mag, angle;
	double bin_value_lo, bin_value_hi;
	const double bins[10] = { 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0 };

	//add into shared memory
	for (i; i < rows * cols; i += stride) {
		int row_idx = i / cols;
		int col_idx = i % cols;

		mag = mag_in[i];
		angle = dir_in[i];

		bin_key = angle / 20;
		bin_key %= 9;
			
		//special case for 180 - move value to 0 bin (bins wrap around)
		if (angle == 180.0) {
			angle = 0;
		}

		//equally divide contributions to different angle bins
		bin_value_lo = ((bins[bin_key + 1] - angle) / 20.0) * mag;
		bin_value_hi = fabs(bin_value_lo - mag);

		//add value to bin
		atomicAdd(&hog_out[(row_idx / 8 * cols * 9) + (col_idx / 8 * 9) + bin_key], bin_value_lo);
		atomicAdd(&hog_out[(row_idx / 8 * cols * 9) + (col_idx / 8 * 9) + ((bin_key+1)%9)], bin_value_lo);
	}
}

void cuda_compute_bins(int n, double *HOG_features, Mat& mag, Mat& dir) {
	const int BYTE_SIZE = mag.rows * mag.cols * sizeof(float);
	const int ARRAY_SIZE = mag.rows * mag.cols;

	float* mag_in, *dir_in;
	cudaMallocManaged(&mag_in, BYTE_SIZE);
	cudaMallocManaged(&dir_in, BYTE_SIZE);

	memcpy(mag_in, (float*) mag.ptr(), BYTE_SIZE);
	memcpy(dir_in, (float*) dir.ptr(), BYTE_SIZE);


	const int numThreads = 1024;
	const int numBlocks = (ARRAY_SIZE + numThreads - 1) / numThreads;

	//improvement: use 2D thread blocks for better synchronization
	group_bin <<< numBlocks, numThreads >>> (n, HOG_features, mag_in, dir_in, mag.rows, mag.cols);
	cudaDeviceSynchronize();

	cudaFree(mag_in);
	cudaFree(dir_in);
	
}

/**********************************************
*				MAIN PROGRAM
***********************************************/
int main() {

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

	cout << image.rows << "\t" << image.cols << "\n" << endl;


	/************************************************************
	*					3. Computing Polar values
	*************************************************************/
	//passover control to GPU

	Mat mag = Mat(img_size, CV_32FC1);
	Mat dir = Mat(img_size, CV_32FC1);

	//call aux function
	compute_gradients(mag, dir, image);

	displayBlock(mag);
	displayBlock(dir);

	//todo: 2x2 blocking of histogram coefficients (into 1x36 coeffs)

	//initialize HOG Features as flattened 3d array
	//indexing: HOG_features[(i * cols * 9) + (j * 9) + k]
	int numFeatures = mag.rows * mag.cols / 64 * 9;
	double* HOG_features;

	cudaMallocManaged(&HOG_features, numFeatures * sizeof(double));
	cudaMemset(HOG_features, 0, numFeatures * sizeof(double));

	//call aux function
	cuda_compute_bins(numFeatures, HOG_features, mag, dir);

	cout << "HOG" << endl;
	for (int i = 0; i < 9; i++) {
		cout << HOG_features[i] << "\t";
	}
	cout << endl;

	//todo: normalization (L2 Norm) of resulting gradients

	cudaFree(HOG_features);

	// cout << dir;
	//todo: display HOG

	return 0;
}