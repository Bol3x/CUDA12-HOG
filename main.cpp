#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


int main() {

	string image_path = "C:\\Users\\ryana\\OneDrive\\Desktop\\dog.jpg";

	//greyscale for now, we can update later
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

	// imshow("Image", image_pad);

	//openCV implementation of Gradient Calculation

	Mat x_grad = Mat(img_size, CV_32FC1);
	Mat y_grad = Mat(img_size, CV_32FC1);

	Sobel(image_pad, x_grad, CV_32FC1, 0, 1, 1);
	Sobel(image_pad, y_grad, CV_32FC1, 1, 0, 1);
	
	Mat x_grad_abs = cv::abs(x_grad);
	Mat y_grad_abs = cv::abs(y_grad);

	Mat mag, dir;
	cartToPolar(x_grad_abs, y_grad_abs, mag, dir, true);

	cout << "\n\n" << dir.at<float>(12,120) << "\n\n";
	// display images
	// imshow("X", x_grad);
	// imshow("Y", y_grad);
	// imshow("Magnitude", mag);

	// Determines how many histograms needed for each cell
	const int ncell_rows = image_pad.rows / 8;
	const int ncell_cols = image_pad.cols / 8;
	const int nBin_size = ncell_rows * ncell_cols;
	
	// Create 9x1 bins for each cell 
	double ***HOGBin = new double**[nBin_size];
	for(int i = 0; i < nBin_size; ++i){
		HOGBin[i] = new double*[nBin_size];
	}

	for(int i = 0; i < ncell_rows; ++i){
		for(int j = 0; j < ncell_cols; ++j){
			HOGBin[i][j] = new double[9] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		}
	}

	
	/*
		Initialize the Bins
		Binning Method : Method 4 
		Source: https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/#h-step-4-calculate-histogram-of-gradients-in-8x8-cells-9x1
		
		Resized image is N x N in size
	*/
	int bin_key = 0; 
	double angle = 0.0, bin_value = 0.0;
	const double bins[9] ={0.0,20.0,40.0,60.0,80.0,100.0,120.0,140.0,160.0};

	for (int row_index = 0; row_index < ncell_rows; ++row_index){
		for(int col_index = 0; col_index < ncell_cols; ++col_index){
			
			// Traverse each cell row-wise
			for(int i = 0; i < dir.rows; i += 8){
				for(int j = 0; j < dir.cols; j+=8){
				
				// Compute for HOG Bin of each cell
				for (int x = i; x < i + 8; ++x){
					for(int y = j; y < j + 8; ++y){
						
						angle = dir.at<float>(x,y);
						bin_key = angle/20;

						bin_value = ((angle - bins[bin_key])/ 20.0) * mag.at<float>(x,y);

						switch(bin_key){
							case 8:
							// If vector direction is between 160 and 180 wrap around 0 
							HOGBin[row_index][col_index][0] += bin_value;
							break;
							
							default:
							HOGBin[row_index][col_index][bin_key + 1] += bin_value;
							break;
						}

						HOGBin[row_index][col_index][bin_key] += mag.at<float>(x,y) - bin_value;
					}
				}

				}
			}
		}
	}

		for(int i = 0; i < ncell_rows; ++i){
			for(int j = 0; j < ncell_cols; ++j){
				for(int k = 0; k < 9; ++k){
					cout << HOGBin[i][j][k] << " ";
				}
				cout << endl;
		}
	}
	//todo: 2x2 blocking of histogram coefficients (into 1x36 coeffs)
	


	//todo: normalization (L2 Norm) of resulting gradients


	// cout << dir ;
	//todo: display HOG

	// Free memory
	for (int i = 0; i < ncell_rows; ++i){
		for(int j = 0; j < ncell_cols; ++j){
			delete[] HOGBin[i][j];
		}
		delete[] HOGBin[i];
	}
	delete[] HOGBin;
	
	cout << "eop" << "\n";

	waitKey(0);
	return 0;
}

//----------Useful References -------------------
//https://learnopencv.com/histogram-of-oriented-gradients/
//https://docs.opencv.org/4.x/d6/d6d/tutorial_mat_the_basic_image_container.html
