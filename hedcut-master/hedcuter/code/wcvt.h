#pragma once

#include <list>
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <set>
#include <fstream>
#include <iostream>
#include <limits>
#include <iomanip>
#include <numeric>
#include <iostream>

#undef check

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/features2d/features2d.hpp>
//#include "opencv2/contrib/contrib.hpp"

//FOR GPU
#ifdef __APPLE__
#include <GLUT/glut.h>
#elif defined(WIN32)
#include <cstdlib>
#include "GL/glut.h"
#else
#include <GL/glut.h>
#endif


extern int argc_GPU;
extern char** argv_GPU;

struct VorCell
{
	VorCell(){}

	VorCell(const VorCell& other)
	{
		site = other.site;
		coverage = other.coverage;
	}

	cv::Point site;
	std::list<cv::Point> coverage;
};

bool compareCell(const std::pair<float, cv::Point>& p1, const std::pair<float, cv::Point>& p2);

struct CVT
{
public:

	CVT()
	{
		iteration_limit = 100;
		max_site_displacement = 1.01f;
		average_termination = false;
		subpixels = 1;
		debug = false;
		size_weight = 100;
		auto_size = false;
		anneal_rate = 1;
		max_weight = 1;
	}

	void compute_weighted_cvt(cv::Mat &  img, std::vector<cv::Point2d> & pts);
	void compute_weighted_cvt_GPU(cv::Mat &  img, std::vector<cv::Point2d> & sites);

	const std::vector<VorCell> & getCells() const 
	{
		return this->cells;
	}

	int iteration_limit;       //max number of iterations when building cvf
	float max_site_displacement; //max tolerable site displacement in each iteration. 
	bool average_termination;
	bool gpu;
	int subpixels;
	float size_weight;	//the amount the size of the cell informs the size of the disk
	bool auto_size;		//modify the disk_size based on the input image and number of disks
	float anneal_rate;	//the amount to multiply the max_site_displacement after each generation
	float max_weight;	//how much to value the maximum displacement over the average

	bool debug;

private:
	static std::vector<VorCell> cells;
	
	void vor(cv::Mat &  img);
	
	
	//For GPU
	void run_GPU(int argc, char**argv, cv::Mat& img);
	void init_GPU(cv::Mat& img);
	static void display_GPU();
	static void vor_GPU();
	static float move_sites_GPU();
	
	//convert a color intensity to distance between 0~1
	inline float color2dist(cv::Mat &  img, cv::Point& p)
	{
		//note: 256 is used here instead of 255 to prevent 0 distance.
		return (256 - img.at<uchar>(p.x, p.y))*1.0f / 256;
	}

	//move the site to the center of its coverage
	inline float move_sites(cv::Mat &  img, VorCell & cell)
	{
		if (cell.coverage.empty()) std::cout << "! Error: cell.coverage " << cell.site << " size = " << cell.coverage.size() << std::endl;

		//Generate virtual high resolution image;
		cv::Size res(img.size().width * subpixels, img.size().height * subpixels);
		cv::Mat resizedImg(res.width, res.height, CV_LOAD_IMAGE_GRAYSCALE);
		cv::resize(img, resizedImg, res, 0, 0, CV_INTER_LINEAR);

		//compute weighted average
		float total = 0;
		cv::Point2d new_pos(0, 0);
		for (auto& c : cell.coverage)
		{
			float d = color2dist(resizedImg, c);
			new_pos.x += d*c.x;
			new_pos.y += d*c.y;
			total += d;
		}

		//normalize
		new_pos.x /= total;
		new_pos.y /= total;

		//Redirect output site to the position in original image
		new_pos.x /= subpixels;
		new_pos.y /= subpixels;

		//update
		float dist = fabs(new_pos.x - cell.site.x/subpixels) + fabs(new_pos.y - cell.site.y/subpixels); //manhattan dist
		cell.site = new_pos;

		//done
		return dist;
	}

	//move the sites to the centers of their coverages
	inline float move_sites(cv::Mat &  img)
	{
		float max_offset = 0;
		float avg_offset = 0;
		for (auto& cell : this->cells)
		{
			float offset = move_sites(img, cell);
			if (offset > max_offset) 
				max_offset = offset;
			avg_offset += offset;
		}
		avg_offset /= this->cells.size();
		if (average_termination) {
			return avg_offset;
		}
		else {
			float off = max_weight * max_offset + (1-max_weight) * avg_offset;
			if (max_weight != 1) {
				std::cout << "Maximum Offset: " << max_offset << ", Average Offset: " << avg_offset << ", Result: " << off << std::endl;
			}
			return off;
		}
	}
};
