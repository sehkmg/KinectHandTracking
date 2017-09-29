#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <string>
using namespace std;
using namespace cv;

void detect(Mat img_8uc1, Mat img_8uc3);

static void help()
{
	printf("\n"
		"This program demonstrated a simple method of connected components clean up of background subtraction\n"
		"When the program starts, it begins learning the background.\n"
		"You can toggle background learning on and off by hitting the space bar.\n"
		"Call\n"
		"./segment_objects [video file, else it reads camera 0]\n\n");
}

int main(int argc, char** argv)
{
	VideoCapture cap;
	bool update_bg_model = true;
	CommandLineParser parser(argc, argv, "{help h||}{@input||}");
	if (parser.has("help"))
	{
		help();
		return 0;
	}
	string input = parser.get<std::string>("@input");
	if (input.empty())
		cap.open(0);
	else
		cap.open(input);
	if (!cap.isOpened())
	{
		printf("\nCan not open camera or video file\n");
		return -1;
	}
	Mat tmp_frame, ycc_frame, out_frame, bwImage;
	cap >> tmp_frame;
	if (tmp_frame.empty())
	{
		printf("can not read data from the video source\n");
		return -1;
	}
	namedWindow("video", 1);
	namedWindow("segmented", 1);
	
	for (;;)
	{
		cap >> tmp_frame;
		if (tmp_frame.empty())
			break;

		cvtColor(tmp_frame, ycc_frame, CV_BGR2YCrCb);
		inRange(ycc_frame, Scalar(0, 133, 77), Scalar(255, 173, 127), out_frame);

		detect(out_frame, tmp_frame);

		imshow("video", tmp_frame);
		imshow("segmented", out_frame);
		char keycode = (char)waitKey(30);
		if (keycode == 27)
			break;
	}
	return 0;
}

void  detect(Mat img_8uc1, Mat img_8uc3)
{
	vector<vector<Point>> contours;
	vector<Point> maxitem;
	double area = 0, areamax = 0;
	int maxn = 0;

	findContours(img_8uc1, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	if (contours.size() > 0)
	{
		for (int i = 0; i < contours.size(); i++)
		{
			area = contourArea(contours[i]);

			if (area>areamax)
			{
				areamax = area;
				maxitem = contours[i];
				maxn = i;
			}
		}
		
		if (areamax>5000)
		{
			approxPolyDP(maxitem, maxitem, 10, 1);

			vector<int> hull;
			vector<Vec4i> defects;

			convexHull(maxitem, hull, false, true);
			
			convexityDefects(maxitem, hull, defects);

			// This cycle marks all defects of convexity of current contours.  
			for (int i = 0; i < defects.size(); i++)
			{
				// Draw marks for all defects.
				line(img_8uc3, maxitem[defects[i][0]], maxitem[defects[i][2]], CV_RGB(255, 255, 0), 1, CV_AA, 0);
				circle(img_8uc3, maxitem[defects[i][2]], 5, CV_RGB(0, 0, 164), 2, 8, 0);
				circle(img_8uc3, maxitem[defects[i][0]], 5, CV_RGB(0, 0, 164), 2, 8, 0);
				line(img_8uc3, maxitem[defects[i][2]], maxitem[defects[i][1]], CV_RGB(255, 255, 0), 1, CV_AA, 0);
			}
		}
	}
}