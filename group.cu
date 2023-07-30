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
	const int stride = blockDim.x * gridDim.x;
	
	float x, y;
	float temp;
	//compute x gradients
	for (i; i < rows * cols; i += stride) {
		//do not include first and last cols (borders)
		y = (i % cols == 0 || i % cols == cols - 1) ?
			0 : input[i + 1] - input[i - 1];

		//do not include first and last rows (borders)
		x = (i < cols || i >= (rows-1) * cols) ?  
			0 : input[i + cols] - input[i - cols];

		mag[i] = sqrt(x * x + y * y);
		temp = atan2(y, x) * 180 / M_PI;
		if (temp < 0) temp += 180;
		dir[i] = temp;
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

	cout << "Copying data to input...\n" << endl;

	//copy matrix data to unified memory
	memcpy(input, input_mat.ptr(), BYTE_SIZE);

	//initialize kernel data
	const int numThreads = 1024;
	const int numBlocks = (ARRAY_SIZE + numThreads - 1) / numThreads;

	cout << "Launching kernel...\n" << endl;

	//launch kernel
	//todo: parallelize kernel executions to improve performance
	computegrad_device <<< numBlocks, numThreads >>> (mag_out, dir_out, input, input_mat.rows, input_mat.cols);
	cudaDeviceSynchronize();

	cout << "Copying mag output into matrices...\n" << endl;

	//copy unified memory into gradient matrices
	memcpy(mag.ptr(), mag_out, BYTE_SIZE);

	cout << "Copying mag output into matrices...\n" << endl;

	//copy unified memory into gradient matrices
	memcpy(dir.ptr(), dir_out, BYTE_SIZE);

	cout << "Freeing memory...\n" << endl;

	//free memory allocated
	cudaFree(mag_out);
	cudaFree(dir_out);
	cudaFree(input);
}

__global__
void group_bin(int n, double* hog_out, float* mag_in, float* dir_in, int rows, int cols) {
	extern __shared__ double temp[]; //globally shared to properly synchronize bin values
	//flattened 3D array
	//see: https://stackoverflow.com/questions/22110663/how-is-a-three-dimensional-array-stored-in-memory
	// https://en.wikibooks.org/wiki/C_Programming/Common_practices
	//temp is arranged by elem-last; ie. temp[0-8] is for block[0][0], temp[9-17] is block[0][1], etc.
	//blocks are arranged in row-major
	//indexing method: i = block row	j = block col	bin_key = bin position
	//access by doing temp[(i*cols*9) + (j*9) + bin_key]

	int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_j = blockIdx.y * blockDim.y + threadIdx.y;

	const int stride_x = blockDim.x * gridDim.x;
	const int stride_y = blockDim.y * gridDim.y;

	//local block index
	int local_idx_x = threadIdx.x % 8;
	int local_idx_y = threadIdx.y % 8;

	//corresponding HOG block bins
	int subblock_i = threadIdx.x / 8;
	int subblock_j = threadIdx.y / 8;

	//initialize HOG feature blocks in shared memory
	//assumes thread blocks of at least 8x8 size
	if (threadIdx.x % 8 > 3 && threadIdx.y % 8 > 3) {
		for (int i = idx_i; i < rows; i += stride_x) {
			for (int j = idx_j; j < cols; j += stride_y) {
					temp[((i + subblock_i) * cols * 9) + ((j + subblock_j) * 9) + (local_idx_x * 3 + local_idx_y)] = 0;
			}
		}
	}
	__syncthreads();


	int bin_key;
	double mag, angle, bin_value_lo, bin_value_hi;
	const double bins[10] = { 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0 };

	//add into shared memory
	for (int i = idx_i; i < rows; i += stride_x) {
		for (int j = idx_j; j < cols; j += stride_y) {
			mag = mag_in[(i * cols) + j];
			angle = dir_in[(i * cols) + j];

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
			temp[(i * cols * 9 + subblock_i) + (j * 9 + subblock_j) + bin_key] += bin_value_lo;
			__syncthreads();
			temp[(i * cols * 9 + subblock_i) + (j * 9 + subblock_j) + ((bin_key + 1) % 9)] += bin_value_hi;
			__syncthreads();
		}
	}

	//writeback to global memory
	if (threadIdx.x % 8 > 3 && threadIdx.y % 8 > 3) {
		for (int i = idx_i; i < rows; i += stride_x) {
			for (int j = idx_j; j < cols; j += stride_y) {
				hog_out[((i + subblock_i) * cols * 9) + ((j + subblock_j) * 9) + (local_idx_x * 3 + local_idx_y)] = 
					temp[((i + subblock_i) * cols * 9) + ((j + subblock_j) * 9) + (local_idx_x * 3 + local_idx_y)];
			}
		}
	}

}

void cuda_compute_bins(int n, double *HOG_features, Mat& mag, Mat& dir) {
	const int BYTE_SIZE = mag.rows * mag.cols * sizeof(float);
	const int ARRAY_SIZE = mag.rows * mag.cols;

	float* mag_in, *dir_in;
	cudaMallocManaged(&mag_in, BYTE_SIZE);
	cudaMallocManaged(&dir_in, BYTE_SIZE);

	memcpy(mag_in, mag.ptr(), BYTE_SIZE);
	memcpy(dir_in, dir.ptr(), BYTE_SIZE);


	//todo
	const int BLOCK_SIZE = 32;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);	//32*32 block (1024 threads)
	dim3 dimGrid(mag.rows / BLOCK_SIZE, mag.cols / BLOCK_SIZE); //create enough blocks to complete array
	

	//rows * cols / 8 = number of blocks
	//9 bins per block, 4 bytes per double elem
	group_bin <<< numBlocks, numThreads , mag.rows * mag.cols / 8 * 9 * sizeof(double) >>> (n, HOG_features, mag_in, dir_in, mag.rows, mag.cols);
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

	const int hog_size = (image.rows / 8 - 1) * (image.cols / 8 - 1) * 36;
	float* hog_features;

	cudaMallocManaged(&hog_features, hog_size * sizeof(float));

	cout << image.rows << "\t" << image.cols << "\n" << endl;


	/************************************************************
	*					3. Computing Polar values
	*************************************************************/
	//passover control to GPU

	Mat mag = Mat(img_size, CV_32FC1);
	Mat dir = Mat(img_size, CV_32FC1);

	//call aux function
	compute_gradients(mag, dir, image);


	//todo: 2x2 blocking of histogram coefficients (into 1x36 coeffs)

	//initialize HOG Features as flattened 3d array
	//indexing: HOG_features[(i * cols * 9) + (j * 9) + k]
	int numFeatures = mag.rows * mag.cols / 8 * 9;
	double* HOG_features = (double*)malloc(sizeof(double) * numFeatures);

	//call aux function
	cuda_compute_bins(numFeatures, HOG_features, mag, dir);


	cout << "HOG features (Block 1)" << endl;
	for (int i = 0; i < 9; i++) {
		cout << HOG_features[i] << "\t";
	}
	cout << endl;

	//todo: normalization (L2 Norm) of resulting gradients




	// cout << dir ;
	//todo: display HOG

	return 0;
}