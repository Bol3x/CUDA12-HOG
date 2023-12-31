#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>


/*
*	compute for the HOG bins of each 8x8 pixel block
*	first, calculate for the gradients
*/
__global__ 
void compute_bins(double* hog_out, unsigned char *input, int rows, int cols) {
	int t_i = blockIdx.x * blockDim.x + threadIdx.x;
	int t_j = blockIdx.y * blockDim.y + threadIdx.y;

	const int stride_i = blockDim.x * gridDim.x;
	const int stride_j = blockDim.y * gridDim.y;

	const double bins[10] = { 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0 };

	//compute x gradients
	for (int i = t_i; i < rows; i += stride_i) {
		for (int j = t_j; j < cols; j += stride_j) 
		{
			//calculate x and y gradients
			//do not include first and last rows (borders)
			double x = (i == 0 || i == rows - 1) ?
				0 : input[((i + 1) * cols) + j] - input[((i - 1) * cols) + j];

			//do not include first and last cols (borders)
			double y = (j == 0 || j == cols - 1) ?
				0 : input[(i * cols) + j + 1] - input[(i * cols) + j - 1];


			double mag = sqrt(x * x + y * y);
			double dir = (atan2(y, x) * 180 / M_PI);
			if (dir < 0) dir += 180;
			if (dir == 180) dir = 0;

			//determine which bins the magnitude will be added to
			int bin_key = dir / 20;
			bin_key %= 9;

			//equally divide contributions to different angle bins
			double bin_value_lo = ((bins[bin_key + 1] - dir) / 20.0) * mag;
			double bin_value_hi = fabs(bin_value_lo - mag);

			//add value to bin
			int out_idx = (((i / 8) * (cols / 8)) + j / 8) * 9;	//output index (flattened index)
			atomicAdd(&hog_out[out_idx + bin_key], bin_value_lo);
			atomicAdd(&hog_out[out_idx + (bin_key + 1) % 9], bin_value_hi);
		}
	}
}

/*
*	copy HOG bin data into a 1xn feature vector
*/
__global__
void copyBinData(double* normHOG, double* HOGBin, int rows, int cols, int n) {
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (t_idx; t_idx < n; t_idx += blockDim.x * gridDim.x) {
		int idx = t_idx / 36;	//get 36 HOG elements per 2x2 block (9x4)
		int elem = t_idx % 36;	//specific element

		//get index of HOG bin to copy
		int i = idx / (cols - 1);
		int j = idx % (cols - 1);

		//copy HOG bins to feature vector
		normHOG[t_idx] = (elem < 18) ?
					HOGBin[(i * cols + j) * 9 + elem] :				//upper row of 2x2 block
					HOGBin[((i + 1) * cols + j) * 9 + (elem - 18)];	//lower row of 2x2 block
	}
}

/*
*	creates the final feature vector from the HOG features in each 8x8 block
*	first copies the bins arranged by 2x2 feature blocks
*	then performs normalization on each of these feature blocks
*/
__global__
void L2norm(double* input, double* norms, int num_norms) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (i; i < num_norms; i += stride) {
		int norm_idx = i / 36;

		if (i % 36 == 0) {
			double sum = 0;

			//could be better... maybe reduction sum without using shared mem (global loc)?
			//problem with shared mem: 
			//only shared within thread blocks; what if 36 elements cross between blocks?
			//then next block will read wrong shared memory location
			#pragma unroll
			for (int j = 0; j < 36; j++) {
				sum += (input[i + j] * input[i + j]);
			}

			norms[norm_idx] = sum;
		}
		__syncthreads();

		input[i] /= sqrt(norms[norm_idx] + 1e-6 * 1e-6);
	}
}