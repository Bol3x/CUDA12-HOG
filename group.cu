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
#include <time.h>

using namespace cv;
using namespace std;

void displayBlock(float *arr, int cols) {
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			cout << arr[(i * cols) + j] << "\t";
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
			int idx = (i * cols) + j;
			//do not include first and last cols (borders)
			y = (j == 0 || j == cols - 1) ?
				0 : input[idx+1] - input[idx-1];

			//do not include first and last rows (borders)
			x = (i == 0 || i == rows - 1) ?
				0 : input[((i + 1) * cols) + j] - input[((i - 1) * cols) + j];

			mag[idx] = sqrt(x * x + y * y);
			temp = atan2(y, x) * 180 / M_PI;
			if (temp < 0) temp += 180;
			dir[idx] = temp;
		}
	}
}

__global__
void cuda_hog_bin(int n, double* hog_out, float* mag_in, float* dir_in, int rows, int cols) {

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	const int stride_i = blockDim.x * gridDim.x;
	const int stride_j = blockDim.y * gridDim.y;

	int bin_key;
	float mag, angle;
	double bin_value_lo, bin_value_hi;
	const double bins[10] = { 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0 };

	//add into shared memory
	for (int x = i; x < rows; x += stride_i) {
		for (int y = j; y < cols; y += stride_j) {

			mag = mag_in[(x * cols) + y];
			angle = dir_in[(x * cols) + y];

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
			atomicAdd(&hog_out[(x / 8 * cols * 9) + (y / 8 * 9) + bin_key], bin_value_lo);
			atomicAdd(&hog_out[(x / 8 * cols * 9) + (y / 8 * 9) + (bin_key + 1) % 9], bin_value_hi);
		}
	}
}

__device__
void L2Normalization(double *HOGFeatures, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	const int stride = blockDim.x * gridDim.x;

	double norm;
	double temp;

	for (i; i < n; i+= stride ){
		norm = 0
		for (j; j < 36; j++){
			temp = HOGFeatures[i+j];
			norm += temp * temp;
		}
		norm = sqrt(norm);

		for (j; j<36; j++){
			HOGFeatures[i+j] /=sqrt(norm*norm + 1e-6*1e-6)
		}
	}
}

__global__ 
void copyBinData(double *HOGFeatures, double *HOGBin, int cols, int num_elem) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_elem) {
        int n = idx / (cols - 1);
        int m = idx % (cols - 1);
        int feature_index = idx * 36;

        for (int j = 0; j < 9; j++) {
            HOGFeatures[feature_index + j] = HOGBin[(n * cols + m) * 9 + j];
            HOGFeatures[feature_index + 9 + j] = HOGBin[(n * cols + m + 1) * 9 + j];
            HOGFeatures[feature_index + 18 + j] = HOGBin[((n + 1) * cols + m) * 9 + j];
            HOGFeatures[feature_index + 27 + j] = HOGBin[((n + 1) * cols + m + 1) * 9 + j];
        }
    }
}

void normalizeGradients(double *HOGFeatures, double ***HOGBin, int rows, int cols, int num_elem) {
    // Flatten HOGBin data into a 1D array for easier copying to the device
    double *flatHOGBin = new double[rows * cols * 9];
    for (int n = 0; n < rows; n++) {
        for (int m = 0; m < cols; m++) {
            for (int i = 0; i < 9; i++) {
                flatHOGBin[(n * cols + m) * 9 + i] = HOGBin[n][m][i];
            }
        }
    }

    // Allocate and copy the flattened HOGBin data to the device
    double *d_flatHOGBin;
   	const int binDataSize = rows * cols * 9 * sizeof(double);
    cudaMallocManaged((void **)&d_flatHOGBin, binDataSize);
    cudaMemcpy(d_flatHOGBin, flatHOGBin, binDataSize, cudaMemcpyHostToDevice);

    // Allocate device memory for HOGFeatures
    double *d_HOGFeatures;
    const int featureSize = num_elem * 36 * sizeof(double);
    cudaMallocManaged((void **)&d_HOGFeatures, featureSize);

    // Define the block and grid dimensions for the kernel
    const int numThreads = 1024;
    const int numBlocks = (num_elem + numThreads - 1) / numThreads;

	cout << "Launching kernels...\n" << endl;

    // Launch the kernel to copy bin data
    copyBinData<<<numBlocks, numThreads>>>(d_HOGFeatures, d_flatHOGBin, cols, num_elem);

    // Wait for all threads to finish
    cudaDeviceSynchronize();

    // Launch the kernel for L2 normalization on each 1x36 feature
    L2Normalization(d_HOGFeatures, num_elem);
	
 	// Wait for all threads to finish
	cudaDeviceSynchronize();

    // Copy the results back from device to host
    cudaMemcpy(HOGFeatures, d_HOGFeatures, featureSize, cudaMemcpyDeviceToHost);

    // Free the allocated device memory
    cudaFree(d_flatHOGBin);
    cudaFree(d_HOGFeatures);
}


/**********************************************
*				MAIN PROGRAM
***********************************************/
int main() {

	clock_t start, end;
	double time_elapsed_grad, time_elapsed_bin;
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

	const int BLOCK_SIZE = 32;
	dim3 threadBlock(BLOCK_SIZE, BLOCK_SIZE);	//32*32 (1024 threads)
	dim3 dimBlock(rows / BLOCK_SIZE, cols / BLOCK_SIZE);

	unsigned char* img_data;
	cudaMallocManaged(&img_data, img_size.area() * sizeof(char));
	memcpy(img_data, image.ptr(), img_size.area() * sizeof(char));


	/************************************************************
	*					3. Computing Polar values
	*************************************************************/
	//passover control to GPU

	int float_byte_size = img_size.area() * sizeof(float);
	float *mag, *dir;

	cudaMallocManaged(&mag, float_byte_size);
	cudaMallocManaged(&dir, float_byte_size);

	start = clock();
	for (int i = 0; i < runs; i++) {
		computegrad_device << < dimBlock, threadBlock >> > (mag, dir, img_data, rows, cols);
		cudaDeviceSynchronize();
	}
	end = clock();

	time_elapsed_grad = ((double)(end - start)) * 1e6 / CLOCKS_PER_SEC / runs;

	cout << "Average time elapsed (in us): " << time_elapsed_grad << endl;

	displayBlock(mag, cols);
	
	displayBlock(dir, cols);


	/************************************************************
	*					4.  HOG Features
	*************************************************************/

	//initialize HOG Features as flattened 3d array
	//indexing: HOG_features[(i * cols * 9) + (j * 9) + k]
	int numFeatures = img_size.area() / 64 * 9;
	double* HOG_features;

	cudaMallocManaged(&HOG_features, numFeatures * sizeof(double));
	cudaMemset(HOG_features, 0, numFeatures * sizeof(double));

	//call aux function
	start = clock();
	for (int i = 0; i < runs; i++) {
		cuda_hog_bin << < dimBlock, threadBlock >> > (numFeatures, HOG_features, mag, dir, rows, cols);
		cudaDeviceSynchronize();
	}
	end = clock();

	time_elapsed_bin = ((double)(end - start)) * 1e6 / CLOCKS_PER_SEC / runs;

	cout << "Average time elapsed (in us): " << time_elapsed_bin << endl;


	cout << "HOG Bins (Block 1)" << endl;
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