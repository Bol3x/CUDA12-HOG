#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void visualizeHOG(const Mat input_image, double*** HOGBin, int ncell_rows, int ncell_cols) {
    // Create a grayscale visualization image with the same size as the input image
    Mat hog_visualization = Mat(input_image.rows, input_image.cols, CV_8UC1, Scalar(0));

    // Calculate the size of each cell in the visualization
    int cell_width = input_image.cols / ncell_cols;
    int cell_height = input_image.rows / ncell_rows;

    int feature_idx = 0;
    for (int r = 0; r < ncell_rows; ++r) {
        for (int c = 0; c < ncell_cols; ++c) {
            // // Calculate the center position of the cell
            // since there is a new block on each cell (overlapping blocks) but the last one
            double center_x = (c + 1) * cell_width;
            double center_y = (r + 1) * cell_height;

            // Get the dominant direction stored in the HOGBin array
            double dominant_direction[4];
            dominant_direction[0] = 0.0;
            dominant_direction[1] = 0.0;
            dominant_direction[2] = 0.0;
            dominant_direction[3] = 0.0;

            int max_bin_idx[4]{ 0,0,0,0 };

            for (int bin_idx = 0; bin_idx < 9; bin_idx++) {
                if (HOGBin[r][c][bin_idx] > dominant_direction[0]) {
                    dominant_direction[2] = dominant_direction[3];
                    max_bin_idx[3] = max_bin_idx[2];

                    dominant_direction[1] = dominant_direction[2];
                    max_bin_idx[2] = max_bin_idx[1];

                    dominant_direction[1] = dominant_direction[0];
                    max_bin_idx[1] = max_bin_idx[0];

                    dominant_direction[0] = HOGBin[r][c][bin_idx];
                    max_bin_idx[0] = bin_idx;
                }
            }

            for (int i = 0; i < 4; ++i) {
                // Calculate the arrow endpoint position based on the dominant gradient direction

                // add
                float currRad = max_bin_idx[i] * 3.14 / 9;
                float dirVecX = cos(currRad);
                float dirVecY = sin(currRad);
                float scale = 0.015; // just a visual_imagealization scale,
                // to see the lines better

// compute line coordinates
                float x1 = center_x - dirVecX * dominant_direction[i] * scale;
                float y1 = center_y - dirVecY * dominant_direction[i] * scale;
                float x2 = center_x + dirVecX * dominant_direction[i] * scale;
                float y2 = center_y + dirVecY * dominant_direction[i] * scale;

                // Draw the arrowed line to represent the dominant gradient direction
                line(hog_visualization,
                    Point(x1, y1),
                    Point(x2, y2),
                    Scalar(255),
                    1, cv::LINE_AA);
            }
            feature_idx++;
        }
    }

    // Show the input image with the HOG visualization overlay
    cv::imshow("HOG Visualization", hog_visualization);
    waitKey(0);
}