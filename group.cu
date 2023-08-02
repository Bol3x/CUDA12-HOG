#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <iostream>
#include <cmath>
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
			int out_idx = (x / 8 * (cols / 8) + y / 8) * 9;
			atomicAdd(&hog_out[out_idx + bin_key], bin_value_lo);
			atomicAdd(&hog_out[out_idx + (bin_key + 1) % 9], bin_value_hi);
		}
	}
}


/*
*	get 2x2 HOG blocks
*/
__global__
void copyBinData(double* normHOG, double* HOGBin, int rows, int cols, int n) {
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (t_idx; t_idx < n; t_idx += blockDim.x * gridDim.x) {
		int idx = t_idx / 36;
		int elem = t_idx % 36;

		int i = idx / cols;
		int j = idx % cols;
		if (i < rows - 1 && j < cols - 1) {
			normHOG[t_idx] = (elem < 18) ? 
				HOGBin[(i * cols + j) * 9 + elem] :			//upper row of 2x2 block
				HOGBin[((i + 1) * cols + j) * 9 + (elem-18)];	//lower row of 2x2 block
		}
	}
}

__global__
void L2norm(double *input, double* norms, int num_norms) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (i; i < num_norms; i += stride) {
		int norm_idx = i / 36;

		if (i % 36 == 0) {
			double sum = 0;

			for (int j = 0; j < 36; j++) {
				sum += (input[i + j] * input[i + j]);
			}
			
			norms[norm_idx] = sum;
		}
		__syncthreads();

		input[i] /= sqrt(norms[norm_idx] + 1e-6 * 1e-6);
	}
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
	*					3. Computing Polar values
	*************************************************************/
	//passover control to GPU

	int float_byte_size = img_size.area() * sizeof(float);
	float *mag, *dir;

	cudaMallocManaged(&mag, float_byte_size);
	cudaMallocManaged(&dir, float_byte_size);

	for (int i = 0; i < runs; i++) {

		start = clock();
		computegrad_device <<< dimBlock, threadBlock >>>(mag, dir, img_data, rows, cols);
		cudaDeviceSynchronize();
		end = clock();

		time_elapsed_grad += ((double)(end - start));
	}

	time_elapsed_grad = time_elapsed_grad * 1e6 / CLOCKS_PER_SEC / runs;

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
	for (int i = 0; i < runs; i++) {
		cudaMemset(HOG_features, 0, numFeatures * sizeof(double));

		start = clock();
		cuda_hog_bin <<< dimBlock, threadBlock >>>(numFeatures, HOG_features, mag, dir, rows, cols);
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

			// Launch the kernel to linearize bin data
			copyBinData <<< numBlocks, numThreads >>> (normHOG, HOG_features, block_rows, block_cols, norm_elems);
			cudaDeviceSynchronize();

			// Launch the kernel for L2 normalization on each 1x36 feature
			L2norm <<< numBlocks, numThreads >>> (normHOG, norm_coeff, norm_elems);
			cudaDeviceSynchronize();

			end = clock();

			time_elapsed_norm += ((double)(end - start));
		}

	time_elapsed_norm = time_elapsed_norm * 1e6 / CLOCKS_PER_SEC / runs;


	/************************************************************
	*					6. HOG Visualization
	*************************************************************/

	cout << "Grad: Average time elapsed (in us): " << time_elapsed_grad << endl;
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