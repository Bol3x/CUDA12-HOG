#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;



int main() {
	string image_path = "C:/Users/Carlo/Downloads/robot.png";

	//greyscale for now
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
	short block_size = 8;

	//pad image to make it divisible by block_size
	Mat image_pad;
	copyMakeBorder(image, image_pad, 
		0, block_size - image.rows % block_size, 
		0, block_size - image.cols % block_size, 
		BORDER_CONSTANT, Scalar(0));

	Size img_size = image_pad.size();

	cout << image_pad.rows << "\t" << image_pad.cols << endl;

	imshow("Image", image_pad);

	//cropped image block location
	Rect crop;
	Mat block;

	int i, j;

	Mat x_grad = Mat(img_size, CV_8SC1);
	Mat y_grad = Mat(img_size, CV_8SC1);


	//loop through each block in the matrix (as per Dalal & Triggs' algorithm)
	for (i = 0; i < image_pad.rows; i += block_size) {
		for (j = 0; j < image_pad.cols; j += block_size) {
			//crop image to block
			block = image_pad(Rect(j, i, block_size, block_size));

			//todo: perform gradient calculation and 
			//store in x_grad and y_grad

		}
	}

	//todo: binning of gradients to angles



	//todo: 2x2 blocking of histogram coefficients (into 1x36 coeffs)
	


	//todo: normalization (L2 Norm) of resulting gradients


	
	//todo: display HOG


	waitKey(0);

	return 0;
}

//https://docs.opencv.org/4.x/d6/d6d/tutorial_mat_the_basic_image_container.html
