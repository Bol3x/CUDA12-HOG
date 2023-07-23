#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace cv;
using namespace std;


/*
* 
*/
__global__
void compute_x_gradients(float* x_out, float* input, int rows, int cols) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;
	
	int i = index;
	//compute x gradients
	//do not include first and last rows (borders)
	for (i; i < (rows) * cols; i += stride) {
		if (i < cols || i >= (rows-1) * cols)
			x_out[i] = 0;
		else
			x_out[i] = input[i + cols] - input[i - cols];
	}
}


__global__
void compute_y_gradients(float* y_out, float* input, int rows, int cols) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;

	int i = index;

	//compute y gradients
	//do not include first and last rows (borders)
	for (int i = index; i < rows * cols; i += stride) {
		if (i % cols == 0 || i % cols == cols - 1)
			y_out[i] = 0;
		else
			y_out[i] = input[i + 1] - input[i - 1];
	}
}


__global__
void compute_magnitudes(float* mag_out, float* x_in, float* y_in, int rows, int cols) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;

	int i = index;

	for (i; i < rows * cols; i += stride) {
		mag_out[i] = sqrt((x_in[i] * x_in[i]) + (y_in[i] * y_in[i]));
	}
}


__global__
void compute_angles(float* dir_out, float* x_in, float* y_in, int rows, int cols, const double pi) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;

	int i = index;
	double temp;

	for (i; i < rows * cols; i += stride) {
		temp = atan2(y_in[i], x_in[i]) * 180/pi;
		if (temp < 0) temp += 180;
		dir_out[i] = temp;
	}
}

__global__
void L2norm_2x2 ( double * x_out, double * x_in, int rows, int cols){
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;

	int i = index;
	double temp;
		for (i; i < rows * cols; i += stride) {
			temp = x_in[i + cols];
			x_out += temp*temp;
		}

}

__global__
void L2norm_sqrt (double * x_out, double * x_in, int n){
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;

	int i = index;
	
	for (i; i< rows; i += stride){
		x_out = sqrt(x_in);
	}
}

__global__
void L2norm (double * x_out, double * x_in, int rows, int cols)
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;

	int i = index;
	for(i; i< rows*cols; i+= stride){
		x_out(i+cols) /= sqrt(x_in*xin + 1e-6*1e-6);
	}

__global__
void normalizeGradients(double *HOGFeatures, double ***HOGBin, int rows, int cols, int num_elem)

	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;

	int i = index;
	int temp = 0;

	for (i; i< (rows-1)*(cols-1)*(9); i+=stride){
		HOGFeatures[temp +i]= 
	}


/*
*	wrapper function to compute gradients using CUDA
*/
void compute_gradients(Mat& x_grad, Mat& y_grad, Mat& input_mat) {

	//number of bytes for the matrix - step is bytes per row
	const int BYTE_SIZE = input_mat.step * input_mat.rows;
	const int ARRAY_SIZE = input_mat.rows * input_mat.cols;

	//create pointers to put into the kernel
	float* x_out, * y_out, * input;

	//allocate unified memory to pointers
	cudaMallocManaged(&input, BYTE_SIZE);
	cudaMallocManaged(&x_out, BYTE_SIZE);
	cudaMallocManaged(&y_out, BYTE_SIZE);

	cout << "Copying data to input...\n" << endl;

	//copy matrix data to unified memory
	memcpy(input, input_mat.ptr(), BYTE_SIZE);

	//initialize kernel data
	const int numThreads = 1024;
	const int numBlocks = (ARRAY_SIZE + numThreads - 1) / numThreads;

	cout << "Launching kernels...\n" << endl;

	//launch kernel
	//todo: parallelize kernel executions to improve performance
	compute_x_gradients <<< numBlocks, numThreads >>> (x_out, input, input_mat.rows, input_mat.cols);
	cudaDeviceSynchronize();

	cout << "Copying x output into matrices...\n" << endl;

	//copy unified memory into gradient matrices
	memcpy(x_grad.ptr(), x_out, BYTE_SIZE);

	compute_y_gradients <<< numBlocks, numThreads >>> (y_out, input, input_mat.rows, input_mat.cols);
	cudaDeviceSynchronize();

	cout << "Copying y output into matrices...\n" << endl;

	//copy unified memory into gradient matrices
	memcpy(y_grad.ptr(), y_out, BYTE_SIZE);

	cout << "Freeing memory...\n" << endl;

	//free memory allocated
	cudaFree(x_out);
	cudaFree(y_out);
	cudaFree(input);
}



/*
*	wrapper function to compute polar values on each element using CUDA
*/
void compute_polar(Mat& mag_mat, Mat& dir_mat, Mat& x_grad, Mat& y_grad) {

	//number of bytes for the matrix - step is bytes per row
	const int BYTE_SIZE = x_grad.step * x_grad.rows;
	const int ARRAY_SIZE = x_grad.rows * x_grad.cols;
	const double PI = atan(1) * 4;

	//create pointers to put into the kernel
	float* x_in, *y_in, *mag, *dir;

	//allocate unified memory to pointers
	cudaMallocManaged(&x_in, BYTE_SIZE);
	cudaMallocManaged(&y_in, BYTE_SIZE);
	cudaMallocManaged(&mag, BYTE_SIZE);
	cudaMallocManaged(&dir, BYTE_SIZE);


	cout << "Copying data to input...\n" << endl;

	//copy matrix data to unified memory
	memcpy(x_in, x_grad.ptr(), BYTE_SIZE);
	memcpy(y_in, y_grad.ptr(), BYTE_SIZE);


	//initialize kernel data
	const int numThreads = 1024;
	const int numBlocks = (ARRAY_SIZE + numThreads - 1) / numThreads;

	cout << "Launching kernels...\n" << endl;

	//launch kernel
	//todo: parallelize kernel executions to improve performance
	compute_magnitudes <<< numBlocks, numThreads >>> (mag, x_in, y_in, x_grad.rows, x_grad.cols);
	cudaDeviceSynchronize();

	compute_angles <<< numBlocks, numThreads >>> (dir, x_in, y_in, x_grad.rows, x_grad.cols, PI);
	cudaDeviceSynchronize();

	cout << "Copying outputs into matrices...\n" << endl;

	//copy unified memory into gradient matrices
	memcpy(mag_mat.ptr(), mag, BYTE_SIZE);
	memcpy(dir_mat.ptr(), dir, BYTE_SIZE);

	cout << "Freeing memory...\n" << endl;

	//free memory allocated
	cudaFree(x_in);
	cudaFree(y_in);
	cudaFree(mag);
	cudaFree(dir);
}


/**********************************************
*				MAIN PROGRAM
***********************************************/
int main() {

	/************************************************************
	*					1. Reading image data
	*************************************************************/

	string image_path = "C:\\Users\\Carlo\\Downloads\\robot.png";

	//greyscale for now, we can update later
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
	short block_size = 8;

	//pad image to make it divisible by block_size
	Mat image_pad;
	copyMakeBorder(image, image_pad,
		0, block_size - image.rows % block_size,
		0, block_size - image.cols % block_size,
		BORDER_CONSTANT, Scalar(0));

	image_pad.convertTo(image_pad, CV_32FC1);

	Size img_size = image_pad.size();

	cout << image_pad.rows << "\t" << image_pad.cols << "\n" << endl;


	/************************************************************
	*					2. Computing Gradients
	*************************************************************/

	Mat x_grad = Mat(img_size, CV_32FC1);
	Mat y_grad = Mat(img_size, CV_32FC1);


	cout << "Computing gradients...\n" << endl;
	compute_gradients(x_grad, y_grad, image_pad);

	cout << "x\n" << endl;
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			cout << x_grad.at<float>(i, j) << "\t";
		}
		cout << endl;
	}
	cout << endl;


	cout << "y\n" << endl;
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			cout << y_grad.at<float>(i, j) << "\t";
		}
		cout << endl;
	}
	cout << endl;

	/************************************************************
	*					3. Computing Polar values
	*************************************************************/

	Mat mag = Mat(img_size, CV_32FC1);
	Mat dir = Mat(img_size, CV_32FC1);

	cout << "Computing polars...\n" << endl;
	compute_polar(mag, dir, x_grad, y_grad); 

	cout << "mag\n" << endl;
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			cout << mag.at<float>(i, j) << "\t";
		}
		cout << endl;
	}
	cout << endl;

	cout << "dir\n" << endl;
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			cout << dir.at<float>(i, j) << "\t";
		}
		cout << endl;
	}
	cout << endl;

	//todo: 2x2 blocking of histogram coefficients (into 1x36 coeffs)



	//todo: normalization (L2 Norm) of resulting gradients




	// cout << dir ;
	//todo: display HOG

	return 0;
}