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
void computegrad_device(float* mag, float* dir, unsigned char* input, int rows, int cols) {
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
		x = (i < cols || i >= (rows - 1) * cols) ?
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
	const int BYTE_SIZE = input_mat.rows * input_mat.cols * sizeof(float);
	const int ARRAY_SIZE = input_mat.rows * input_mat.cols;

	//create pointers to put into the kernel
	float* mag_out, * dir_out;
	unsigned char* input;

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


	cout << "mag" << endl;
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			cout << mag.at<float>(i, j) << "\t";
		}
		cout << endl;
	}
	cout << endl;

	cout << "dir" << endl;
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			cout << dir.at<float>(i, j) << "\t";
		}
		cout << endl;
	}


	/************************************************************
	*					3. Computing Polar values
	*************************************************************/


	//todo: 2x2 blocking of histogram coefficients (into 1x36 coeffs)



	//todo: normalization (L2 Norm) of resulting gradients




	// cout << dir ;
	//todo: display HOG

	return 0;
}