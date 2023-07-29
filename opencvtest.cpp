#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

  #include <windows.h>


int main() {
	// 
	char buffer[MAX_PATH];
  	::GetCurrentDirectory(MAX_PATH, buffer);
  	cout << "Current directory: " << buffer << endl;
	string image_path = ".\\input_img\\dog.jpg";

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

	imshow("Image", image_pad);

	//openCV implementation of Gradient Calculation

	Mat x_grad = Mat(img_size, CV_32FC1);
	Mat y_grad = Mat(img_size, CV_32FC1);

	Sobel(image_pad, x_grad, CV_32FC1, 0, 1, 1);
	Sobel(image_pad, y_grad, CV_32FC1, 1, 0, 1);
	
	Mat x_grad_abs = cv::abs(x_grad);
	Mat y_grad_abs = cv::abs(y_grad);

	Mat mag, dir;
	cartToPolar(x_grad_abs, y_grad_abs, mag, dir, true);

	//debug: display images
	//imshow("X", x_grad);
	//imshow("Y", y_grad);
	//imshow("Magnitude", mag);

	cout << dir << endl;

	// int i, j;

	//loop through each block in the matrix (as per Dalal & Triggs' algorithm)
	// for (i = 0; i < image_pad.rows; i += block_size) {
	// 	for (j = 0; j < image_pad.cols; j += block_size) {

	// 		//todo: binning of gradients to angles


	// 	}
	// }


	//todo: 2x2 blocking of histogram coefficients (into 1x36 coeffs)
	


	//todo: normalization (L2 Norm) of resulting gradients


	
	//todo: display HOG
	Mat HOG_image;

	//unsure for these values
	int scaleFactor = 1;
	double viz_factor = 10;

	//resize HOG_image to be the same as image
	resize(image,HOG_image, Size(image.cols, image.rows));

	int gradientBinSize = 9;
	float radiansRange = 3.14/(float)gradientBinSize;

	//prepare data structure: 9 orientation/gradient strenghts for each cell

	//typical values for winSize and cellSize
	Size winSize = Size(128,64); //size of image cropped to multiple of the cell size
	Size cellSize = Size(8,8); 

	int xDirectionCells = winSize.width / cellSize.width;
	int yDirectionCells = winSize.height / cellSize.height;
	int totalCells = xDirectionCells * yDirectionCells;
	float*** gradientStrengths = new float**[yDirectionCells];
    int** cellUpdateCounter   = new int*[yDirectionCells];

    for (int y=0; y<yDirectionCells; y++)
    {
        gradientStrengths[y] = new float*[xDirectionCells];
        cellUpdateCounter[y] = new int[xDirectionCells];
        for (int x=0; x<xDirectionCells; x++)
        {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;
 
            for (int bin=0; bin<gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }
 
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int xDirectionBlocks = xDirectionCells - 1;
    int yDirectionBlocks = yDirectionCells - 1;
 
    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int xCell = 0;
    int yCell = 0;
 
    for (int xBlock=0; xBlock<xDirectionBlocks; xBlock++)
    {
        for (int yBlock=0; yBlock<yDirectionBlocks; yBlock++)            
        {
            // 4 cells per block ...
            for (int nthCell=0; nthCell<4; nthCell++)
            {
                // compute corresponding cell nr
                int xCell = xBlock;
                int yCell = yBlock;
                if (nthCell==1) yCell++;
                if (nthCell==2) xCell++;
                if (nthCell==3)
                {
                    xCell++;
                    yCell++;
                }
 
                for (int bin=0; bin<gradientBinSize; bin++)
                {
                    float gradientStrength = descriptorValues[ descriptorDataIdx ];
                    descriptorDataIdx++;
 
                    gradientStrengths[yCell][xCell][bin] += gradientStrength;
 
                } // for (all bins)
 
 
                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cell was updated,
                // to compute average gradient strengths
                cellUpdateCounter[yCell][xCell]++;
 
            } // for (all cells)
 
 
        } // for (all block x pos)
    } // for (all block y pos)
 
 
    // compute average gradient strengths
    for (int yCell=0; yCell<yDirectionCells; yCell++)
    {
        for (int xCell=0; xCell<yDirectionCells; xCell++)
        {
 
            float NrUpdatesForThisCell = (float)cellUpdateCounter[yCell][xCell];
 
            // compute average gradient strenghts for each gradient bin direction
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                gradientStrengths[yCell][xCell][bin] /= NrUpdatesForThisCell;
            }
        }
    }
 
 
    cout << "descriptorDataIdx = " << descriptorDataIdx << endl;
 
    // draw cells
    for (int yCell=0; yCell<yDirectionCells; yCell++)
    {
        for (int xCell=0; xCell<xDirectionCells; xCell++)
        {
            int xDraw = xCell * cellSize.width;
            int yDraw = yCell * cellSize.height;
 
            int mx = xDraw + cellSize.width/2;
            int my = yDraw + cellSize.height/2;
 
            rectangle(visual_image,
                      Point(xDraw*scaleFactor,yDraw*scaleFactor),
                      Point((xDraw+cellSize.width)*scaleFactor,
                      (yDraw+cellSize.height)*scaleFactor),
                      CV_RGB(100,100,100),
                      1);
 
            // draw in each cell all 9 gradient strengths
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                float currentGradStrength = gradientStrengths[yCell][xCell][bin];
 
                // no line to draw?
                if (currentGradStrength==0)
                    continue;
 
                float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;
 
                float dirVecX = cos( currRad );
                float dirVecY = sin( currRad );
                float maxVecLen = cellSize.width/2;
                float scale = viz_factor; // just a visual_imagealization scale,
                                          // to see the lines better
 
                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
 
                // draw gradient visual_imagealization
                line(HOG_image,
                     Point(x1*scaleFactor,y1*scaleFactor),
                     Point(x2*scaleFactor,y2*scaleFactor),
                     CV_RGB(0,0,255),
                     1);
 
            } // for (all bins)
 
        } // for (xCell)
    } // for (yCell)
 
 
    // don't forget to free memory allocated by helper data structures!
    for (int y=0; y<yDirectionCells; y++)
    {
      for (int x=0; x<xDirectionCells; x++)
      {
           delete[] gradientStrengths[y][x];            
      }
      delete[] gradientStrengths[y];
      delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;

	waitKey(0);

	return 0;
}

//----------Useful References -------------------
//https://learnopencv.com/histogram-of-oriented-gradients/
//https://docs.opencv.org/4.x/d6/d6d/tutorial_mat_the_basic_image_container.html
//https://web.archive.org/web/20140822212002/http://www.juergenwiki.de/work/wiki/doku.php?id=public:hog_descriptor_computation_and_visualization
