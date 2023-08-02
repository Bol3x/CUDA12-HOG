#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

#define _USE_MATH_DEFINES
#include <cmath>


using namespace cv;
using namespace std;


/**
*	Computes for the x and y gradients of a given input matrix
*	Performs a 1D sobel filter with the top and left values negative, while bottom and right values positive
*	Stores corresponding gradient value in x_grad and y_grad matrices
*/
void compute_gradients(Mat x_grad, Mat y_grad, Mat input_data) {
	//initialize first row as all 0's
	for (int j = 0; j < input_data.cols; j++)
		x_grad.at<float>(0, j) = 0;

	//compute x gradients
	float top, bottom;
	for (int i = 1; i < input_data.rows - 1; i++) {
		for (int j = 0; j < input_data.cols; j++) {
			bottom = input_data.at<float>(i + 1, j);
			top = input_data.at<float>(i - 1, j);

			x_grad.at<float>(i, j) = bottom - top;
		}
	}

	//initialize first column as all 0's
	for (int i = 0; i < input_data.rows; i++)
		y_grad.at<float>(i, 0) = 0;

	//compute y gradients
	float left, right;
	for (int i = 0; i < input_data.rows; i++) {
		for (int j = 1; j < input_data.cols - 1; j++) {
			right = input_data.at<float>(i, j + 1);
			left = input_data.at<float>(i, j - 1);

			y_grad.at<float>(i, j) = right - left;
		}
	}
}


/*
*	Computes the polar representation of the x and y magnitudes of each element in the 2 matrices.
*	Returns the magnitude and direction of each element and stores them in mag and dir respectively
*/
void compute_polar(Mat mag, Mat dir, Mat x_mag, Mat y_mag) {
	float x_val, y_val, res_dir;
	double pi = atan(1) * 4;
	//compute magnitude and direction of cartesian coordinates x_val and y_val
	for (int i = 0; i < x_mag.rows; i++) {
		for (int j = 0; j < x_mag.cols; j++) {
			x_val = x_mag.at<float>(i, j);
			y_val = y_mag.at<float>(i, j);

			mag.at<float>(i, j) = sqrt(pow(x_val, 2) + pow(y_val, 2));
			res_dir = atan2(y_val, x_val) * 180/pi;
			
			//if resulting direction is < 0, negate by adding 180
			if (res_dir < 0) res_dir += 180;
			dir.at<float>(i, j) = res_dir;
		}
	}
}


void bin_gradients(double*** HOGBin, Mat mag, Mat dir) {
	/*
	Initialize the Bins
	Binning Method : Method 4
	Reference: https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/#h-step-4-calculate-histogram-of-gradients-in-8x8-cells-9x1

	* bins are are stored sequentially - index 0 = bin 0 and index 8 = bin 160
	* Magnitude of the corresponding pixel is distributed
*/
	int bin_key = 0;
	int bin_key_hi = 0, bin_key_low = 0;
	double mag_val = 0.0, angle = 0.0, bin_value_lo = 0.0, bin_value_hi = 0.0;
	const double bins[10] = { 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0 };
	
	int HOG_row = 0, HOG_col = 0;

	// Traverse each cell in the original data
	for (int i = 0; i < dir.rows; i++) {
		for (int j = 0; j < dir.cols; j++) {
			//get corresponding HOG bin block
			HOG_row = i / 8;
			HOG_col = j / 8;

			//angle 
			angle = dir.at<float>(i, j);
			//mag
			mag_val = mag.at<float>(i, j);

			// Round down to get which bin the direction belong to.
			bin_key = angle / 20;
			bin_key %= 9;

			bin_key_hi = bins[bin_key+1];
			bin_key_low = bins[bin_key];

			//special case for 180 - move value to 0 bin (bins wrap around)
			if (angle >= 180.0) {
				angle = 0;
			}

			//equally divide contributions to different angle bins
			bin_value_lo = ((bin_key_hi- angle) / 20.0) * mag_val;
			bin_value_hi = ((bin_key_low - angle)/ 20.0) * mag_val;

			//add value to bin
			HOGBin[HOG_row][HOG_col][bin_key] += bin_value_lo;
			HOGBin[HOG_row][HOG_col][(bin_key + 1) % 9] += bin_value_hi;
		}
	}
}

void L2Normalization(double *HOGFeatures, int feature_index){
	int start = feature_index - 36;
	double k = 0;

	// Get sum of squares
	for (int i = start; i < feature_index; ++i){
		k += pow(HOGFeatures[i], 2);
	}

	k = sqrt(k); 


	// Divide each element from the resulting value derived from k
	for (int i = start; i < feature_index; ++i){
		HOGFeatures[i] = HOGFeatures[i] /k;
	}
}

void normalizeGradients(double *HOGFeatures, double ***HOGBin, int rows, int cols){
	int feature_index = 0;
		for (int i = 0; i < rows -1 ; i++) {
			for (int j = 0; j < cols - 1; j++) {

				for (int x = i; x < i + 2; ++x){
					for(int y = j; y < j + 2; ++y){

						for (int k = 0; k < 9; k++){
							// Append the histogram of each cell
							HOGFeatures[feature_index++] = HOGBin[x][y][k];
						}
				}
			}
			// Performs L2 normalization from index (feature_index - 36) to feature_index based on the current value in the loop
			L2Normalization(HOGFeatures, feature_index);
		}
	}
	
}

void visualizeHOG(const cv::Mat& input_image,double * HOGFeatures, int ncell_rows, int ncell_cols) {
    // Create a blank image with the same size as the input image
    cv::Mat hog_visualization = input_image.clone();

    // Calculate the size of each cell in the visualization
    int cell_width = hog_visualization.cols / ncell_cols;
    int cell_height = hog_visualization.rows / ncell_rows;

    // Draw HOG visualization
    int bin_idx = 0;
    for (int r = 0; r < ncell_rows; ++r) {
        for (int c = 0; c < ncell_cols; ++c) {
            double feature_value = HOGFeatures[bin_idx]; // Get the HOG feature value for the cell

            // grayscale
            cv::Scalar color(255 * feature_value, 255 * feature_value, 255 * feature_value);

            // Draw a rectangle on the HOG visualization image to represent the cell
            cv::rectangle(hog_visualization,
                          cv::Point(c * cell_width, r * cell_height),
                          cv::Point((c + 1) * cell_width, (r + 1) * cell_height),
                          color, -1);

            bin_idx++; // Move to the next bin
        }
    }

    // Overlay the HOG visualization on top of the input image
    double alpha = 0.1; // You can adjust the alpha value to control the transparency of the overlay
    cv::addWeighted(hog_visualization, alpha, input_image, 1.0 - alpha, 0.0, hog_visualization);

    // Show the input image with the HOG visualization overlay
    cv::imshow("HOG Visualization", hog_visualization);
    cv::waitKey(0);
}

// void visualizeHOG(double* HOGFeatures,  int ncell_rows, int ncell_cols) {
//     // Image size for visualization
//     const int img_width = ncell_cols * 10; // Each cell will be 10 pixels wide
//     const int img_height = ncell_rows * 10; // Each cell will be 10 pixels high

//     // Create a blank image
//     cv::Mat hog_image(img_height, img_width, CV_8UC3, cv::Scalar(255, 255, 255));

//     // Draw HOG visualization
//     int bin_idx = 0;
//     for (int r = 0; r < ncell_rows; ++r) {
//         for (int c = 0; c < ncell_cols; ++c) {
//             float feature_value = HOGFeatures[bin_idx]; // Get the HOG feature value for the cell

//             // Use color maps or grayscale to represent the gradient direction and magnitude.
//             // For simplicity, I'll just use grayscale here.
//             cv::Scalar color(255 * feature_value, 255 * feature_value, 255 * feature_value);
//             cv::rectangle(hog_image, cv::Point(c * 10, r * 10), cv::Point((c + 1) * 10, (r + 1) * 10), color, -1);

//             bin_idx++; // Move to the next bin
//         }
//     }

//     // Show the HOG visualization
//     cv::imshow("HOG Visualization", hog_image);
//     cv::waitKey(0);
// }

double *getHOGFeatures(Size img_size, Mat img_pad, double ***HOGBin){

	
	/************************************************************
	*					2. Computing Gradients
	*************************************************************/

	Mat x_grad = Mat(img_size, CV_32FC1);
	Mat y_grad = Mat(img_size, CV_32FC1);

	
	//initialize first row as all 0's
	for (int j = 0; j < img_pad.cols; j++)
		x_grad.at<float>(0, j) = 0;

	//compute x gradients
	float top, bottom;
	for (int i = 1; i < img_pad.rows - 1; i++) {
		for (int j = 0; j < img_pad.cols; j++) {
			bottom = img_pad.at<float>(i + 1, j);
			top = img_pad.at<float>(i - 1, j);

			x_grad.at<float>(i, j) = bottom - top;
		}
	}

	//initialize first column as all 0's
	for (int i = 0; i < img_pad.rows; i++)
		y_grad.at<float>(i, 0) = 0;
		

	//compute y gradients
	float left, right;
	for (int i = 0; i < img_pad.rows; i++) {
		for (int j = 1; j < img_pad.cols - 1; j++) {
			right = img_pad.at<float>(i, j + 1);
			left = img_pad.at<float>(i, j - 1);

			y_grad.at<float>(i, j) = right - left;
		}
	}
	
	/************************************************************
	*					3. Computing Polar values
	*************************************************************/


	Mat mag = Mat(img_size, CV_32FC1); 
	Mat dir = Mat(img_size, CV_32FC1);

	float x_val, y_val, res_dir;
	double pi = atan(1) * 4;
	
	//compute magnitude and direction of cartesian coordinates x_val and y_val
	for (int i = 0; i < x_grad.rows; i++) {
		for (int j = 0; j < x_grad.cols; j++) {
			x_val = x_grad.at<float>(i, j);
			y_val = y_grad.at<float>(i, j);

			mag.at<float>(i, j) = sqrt(pow(x_val, 2) + pow(y_val, 2));
			res_dir = atan2(y_val, x_val) * 180/pi;
			
			//if resulting direction is < 0, negate by adding 180
			if (res_dir < 0) res_dir += 180;
			dir.at<float>(i, j) = res_dir;
		}
	}

	/************************************************************
	*			4. Binning Gradients to Angle bins
	*************************************************************/

	// Determines how many blocks are needed for the image
	const int ncell_rows = img_pad.rows / 8;
	const int ncell_cols = img_pad.cols / 8;
	const int nBin_size = ncell_rows * ncell_cols;
	

	/*
	Initialize the Bins
	Binning Method : Method 4
	Reference: https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/#h-step-4-calculate-histogram-of-gradients-in-8x8-cells-9x1

	* bins are are stored sequentially - index 0 = bin 0 and index 8 = bin 160
	* Magnitude of the corresponding pixel is distributed
	*/
	int bin_key = 0;
	int bin_key_hi = 0, bin_key_low = 0;
	double mag_val = 0.0, angle = 0.0, bin_value_lo = 0.0, bin_value_hi = 0.0;
	const double bins[10] = { 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0 };
	
	int HOG_row = 0, HOG_col = 0;

	// Traverse each cell in the original data
	for (int i = 0; i < dir.rows; i++) {
		for (int j = 0; j < dir.cols; j++) {
			//get corresponding HOG bin block
			HOG_row = i / 8;
			HOG_col = j / 8;

			//angle 
			angle = dir.at<float>(i, j);
			//mag
			mag_val = mag.at<float>(i, j);

			// Round down to get which bin the direction belong to.
			bin_key = angle / 20;
			bin_key %= 9;

			bin_key_hi = bins[bin_key+1];
			bin_key_low = bins[bin_key];

			//special case for 180 - move value to 0 bin (bins wrap around)
			if (angle >= 180.0) {
				angle = 0;
			}

			//equally divide contributions to different angle bins
			bin_value_lo = ((bin_key_hi- angle) / 20.0) * mag_val;
			bin_value_hi = ((bin_key_low - angle)/ 20.0) * mag_val;

			//add value to bin
			HOGBin[HOG_row][HOG_col][bin_key] += bin_value_lo;
			HOGBin[HOG_row][HOG_col][(bin_key + 1) % 9] += bin_value_hi;
		}
	}

	/************************************************************
	*			4. Normalize Gradients in a 16x16 cell
	*************************************************************/

	// Determine number of features
	
	const int features = (ncell_rows - 1) * (ncell_cols - 1) * 36;
	double *HOGFeatures = new double[features];

		int feature_index = 0;
		for (int i = 0; i < ncell_rows -1 ; i++) {
			for (int j = 0; j < ncell_cols - 1; j++) {

				for (int x = i; x < i + 2; ++x){
					for(int y = j; y < j + 2; ++y){

						for (int k = 0; k < 9; k++){
							// Append the histogram of each cell
							HOGFeatures[feature_index++] = HOGBin[x][y][k];
						}
				}
			}
			// Performs L2 normalization from index (feature_index - 36) to feature_index based on the current value in the loop
				int start = feature_index - 36;
				double k = 0;

				// Get sum of squares
				for (int i = start; i < feature_index; ++i){
					k += pow(HOGFeatures[i], 2);
				}

				k = sqrt(k); 


				// Divide each element from the resulting value derived from k
				for (int i = start; i < feature_index; ++i){
					HOGFeatures[i] = HOGFeatures[i] /k;
				}
		}
	}
	

	return HOGFeatures;
}

int main() {

	/************************************************************
	*					1. Reading image data
	*************************************************************/

	string image_path = "D:/Github_Repositories/CUDA12-HOG/input_img/dog.jpg";

	//greyscale for now, we can update later
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
	short block_size = 8;

	imshow("Image", image);
	//pad image to make it divisible by block_size
	Mat image_pad;
	copyMakeBorder(image, image_pad, 
		0, block_size - image.rows % block_size, 
		0, block_size - image.cols % block_size, 
		BORDER_CONSTANT, Scalar(0));

	image_pad.convertTo(image_pad, CV_32FC1);

	Size img_size = image_pad.size();

	// cout << image_pad.rows << "\t" << image_pad.cols << "\n" << endl;

	const int ncell_rows = image_pad.rows / 8;
	const int ncell_cols = image_pad.cols / 8;

		// initialize 9x1 bins corresponding to each block on the image
		double ***HOGBin = new double**[ncell_rows];
		for(int i = 0; i < ncell_rows; ++i){
			HOGBin[i] = new double* [ncell_cols];
			for (int j = 0; j < ncell_cols; ++j) {
				HOGBin[i][j] = new double[9] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
			}
		}

 
	double* HOGFeatures = getHOGFeatures(img_size,image_pad, HOGBin);
	const int features = (ncell_rows - 1) * (ncell_cols - 1) * 36;

	for(int i = 0; i < features; ++i){
		cout << HOGFeatures[i] << "\n";
	}
	visualizeHOG(image_pad,HOGFeatures, ncell_rows, ncell_cols);
	// Free memory
	
	delete[] HOGFeatures; 
	waitKey(0);
	return 0;
}

//----------Useful References -------------------
//https://learnopencv.com/histogram-of-oriented-gradients/
//https://docs.opencv.org/4.x/d6/d6d/tutorial_mat_the_basic_image_container.html
//https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/#h-step-4-calculate-histogram-of-gradients-in-8x8-cells-9x1
