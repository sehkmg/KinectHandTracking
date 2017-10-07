#include "NtKinect.h"

using namespace cv;

double getBendDeg(Mat hand_bwImg, Mat rgbImg);
void markerPos(Mat bwImg, Mat depthImg, DepthSpacePoint &center, UINT16 &depth);

int main()
{
	NtKinect kinect;
	
	CameraSpacePoint handSp;		//Stores position of a hand.
	double bendDeg;				//Stores bending degree of a hand.

	/* Demonstration code */
	namedWindow("rgb", 1);
	namedWindow("hand_segment", 1);
	namedWindow("infrared", 1);
	namedWindow("depth", 1);
	/* Demonstration code */
	
	while(1)
	{
		kinect.setRGB();
		kinect.setInfrared();
		kinect.setDepth(false);

		Mat ycc_frame, hand_bwImg;
		//change frame to black/white.
		//(skin color->white, background->black)
		cvtColor(kinect.rgbImage, ycc_frame, CV_BGR2YCrCb);
		inRange(ycc_frame, Scalar(0, 133, 77), Scalar(255, 173, 127), hand_bwImg);

		//Calculate bending degree of a hand.
		bendDeg = getBendDeg(hand_bwImg, kinect.rgbImage);

		Mat marker_bwImg;
		inRange(kinect.infraredImage, Scalar(65530), Scalar(65535), marker_bwImg);	//make bw image such that marker: white, others: black.

		DepthSpacePoint dp;
		UINT16 d;
		markerPos(marker_bwImg, kinect.depthImage, dp, d);	//Get depth and position of a marker.
								
		kinect.coordinateMapper->MapDepthPointToCameraSpace(dp, d, &handSp);	//Convert a depth space point to a camera space point.

		/* Demonstration code */
		String bend = "Bend degree: " + to_string((int)bendDeg);
		putText(kinect.rgbImage, bend, Point(30, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(34, 177, 76), 2);

		String pos = "X: " + to_string((int)(handSp.X * 1000)) + " Y: " + to_string((int)(handSp.Y * 1000)) + " Z: " + to_string((int)(handSp.Z * 1000));
		putText(kinect.depthImage, pos, Point(30, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(50000), 2);

		imshow("rgb", kinect.rgbImage);
		imshow("hand_segment", hand_bwImg);
		imshow("infrared", kinect.infraredImage);
		imshow("depth", kinect.depthImage);
		/* Demonstration code */

		char keycode = (char)waitKey(30);
		if (keycode == 27)
			break;
	}
	cv::destroyAllWindows();
	return 0;
}

//Calculate bending degree of a hand.
//Here, hand should be white in hand_bwImg.
double getBendDeg(Mat hand_bwImg, Mat rgbImg)
{
	double bendDeg = 0;

	vector<vector<Point>> contours;
	vector<Point> maxitem;
	double area = 0, areamax = 0;

	//find contours in black/white image.
	findContours(hand_bwImg, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	if (contours.size() > 0)
	{
		//choose the contour that has max area.
		for (int i = 0; i < contours.size(); i++)
		{
			area = contourArea(contours[i]);

			if (area>areamax)
			{
				areamax = area;
				maxitem = contours[i];
			}
		}
		
		if (areamax>5000)
		{
			//approximate the contour to polygon.
			approxPolyDP(maxitem, maxitem, 10, 1);

			vector<int> hull;
			vector<Vec4i> defects;

			//find convex hull.
			convexHull(maxitem, hull, false, true);
			
			//find convexity defects.
			convexityDefects(maxitem, hull, defects);

			//Calculate bending degree of a hand by averaging depth of convexity defects.
			for (int i = 0; i < defects.size(); i++)
			{
				bendDeg += defects[i][3];

				/* Demonstration code */
				line(rgbImg, maxitem[defects[i][0]], maxitem[defects[i][2]], CV_RGB(255, 255, 0), 1, CV_AA, 0);
				circle(rgbImg, maxitem[defects[i][2]], 5, CV_RGB(0, 0, 164), 2, 8, 0);
				circle(rgbImg, maxitem[defects[i][0]], 5, CV_RGB(0, 0, 164), 2, 8, 0);
				line(rgbImg, maxitem[defects[i][2]], maxitem[defects[i][1]], CV_RGB(255, 255, 0), 1, CV_AA, 0);
				/* Demonstration code */
			}
			bendDeg /= defects.size();
			bendDeg /= 256.0;
		}
	}
	return bendDeg;
}

//Get depth and position of a marker.
//Here, marker should be white in bwImg.
//**CAUTION: The resolution of bwImg and depthImg should be same.
void markerPos(Mat bwImg, Mat depthImg, DepthSpacePoint &center, UINT16 &depth)
{
	vector<vector<Point>> contours;
	vector<Point> maxitem;
	double area = 0, areamax = 0;

	//find contours in black/white image.
	findContours(bwImg, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	if (contours.size() > 0)
	{
		//choose the contour that has max area.
		for (int i = 0; i < contours.size(); i++)
		{
			area = contourArea(contours[i]);

			if (area > areamax)
			{
				areamax = area;
				maxitem = contours[i];
			}
		}

		if (areamax > 100)
		{
			//approximate the contour to polygon.
			approxPolyDP(maxitem, maxitem, 5, 1);

			Rect boundRect;
			Point ul, dr;

			double depth_avg = 0;
			int count = 0;

			//find bounding rectangle of approximated polygon.
			boundRect = boundingRect(Mat(maxitem));
			ul = boundRect.tl();	//up left point of the rectangle.
			dr = boundRect.br();	//down right point of the rectangle.

									//calculate the center of the marker.
			center.X = (ul.x + dr.x) / 2;
			center.Y = (ul.y + dr.y) / 2;

			//make the rectangle bigger than the marker. (we want to calculate average depth around the marker.)
			ul.x = ul.x - min(ul.x, 10);
			ul.y = ul.y - min(ul.y, 10);
			dr.x = dr.x + min(512 - dr.x, 10);
			dr.y = dr.y + min(424 - dr.y, 10);

			//calculate average depth around the marker.
			for (int y = 0; y < depthImg.rows; y++)
			{
				for (int x = 0; x < depthImg.cols; x++)
				{
					if (ul.x <= x && x <= dr.x && ul.y <= y && y <= dr.y)
						if (depthImg.at<UINT16>(y, x) != 0)		//depth sensor can't detect depth of retro-reflective material. dismiss the depth of the marker.
						{
							depth_avg += depthImg.at<UINT16>(y, x);
							count++;
						}
				}
			}
			depth_avg = depth_avg / count;
			depth_avg = depth_avg * 4500 / 65535;
			depth = (UINT16)depth_avg;

			/* Demonstration code */
			rectangle(depthImg, ul, dr, Scalar(30000), 2, 8, 0);
			circle(depthImg, ul, 5, Scalar(30000), 2, 8, 0);
			circle(depthImg, dr, 5, Scalar(30000), 2, 8, 0);
			/* Demonstration code */
		}
	}
}