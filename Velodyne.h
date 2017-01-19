/*

The GNU General Public License

Copyright (C) 2017 Yung Feng Chang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#if defined(_MSC_VER) && _MSC_VER >= 1900
#pragma warning(push) 
#pragma warning(disable:4996) 
#endif 

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector> 
#include <chrono> 
#include <algorithm>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <boost/make_shared.hpp>
#include <pcl/point_representation.h>
#include <pcl/filters/filter.h>

using namespace std;
using namespace pcl;

// Namespace for visualizer 
// using visualization::PointCloudColorHandlerGenericField;
// using visualization::PointCloudColorHandlerCustom;

#define MAX_FRAME 600

// Typedef to make expression more concise 
typedef PointNormal PointNormalT;
typedef PointCloud<PointNormalT> PointCloudWithNormals;
typedef chrono::high_resolution_clock Clock;

// Global variables for visualizer 
// pcl::visualization::PCLVisualizer *p;
// int vp_1, vp_2;

// Load .bin files and return point cloud object
PointCloud<PointXYZI>::Ptr MyFileOpen(const string& filename)
{
	// Input binary files
	ifstream input(filename, ios::in | ios::binary);
	if (!input.good()) {
		exit(EXIT_FAILURE);
	}
	input.seekg(0, ios::beg);

	PointCloud<PointXYZI>::Ptr points(new pcl::PointCloud<PointXYZI>);

	// The format of Velodyne data is (x , y , z , reflection) per line in .bin file.
	int i;
	for (i = 0; input.good() && !input.eof(); i++) {
		PointXYZI point;
		input.read((char *)&point.x, 3 * sizeof(float));
		input.read((char *)&point.intensity, sizeof(float));
		points->push_back(point);
	}
	input.close();
	return points;
}

// Output Homogeneous matrices to text file
void MyFileOutput_Mat(const vector<Eigen::Matrix4f>& output)
{
	ofstream OutPut_File;
	OutPut_File.open("OutPut_H.txt");
	for (auto& row : output)
	{
		OutPut_File << row;
		OutPut_File << '\n';
	}
	OutPut_File.close();
}

struct PCD
{
	PointCloud<PointXYZ>::Ptr cloud;
	string f_name;

	PCD() : cloud(new PointCloud<PointXYZ>) {};
};

// Define comparator as a functor
struct PCDComparator
{
	bool operator () (const PCD& p1, const PCD& p2)
	{
		return (p1.f_name < p2.f_name);
	}
};

// Define a new point representation for < x, y, z, curvature >
class MyPointRepresentation : public PointRepresentation <PointNormalT>
{
	using PointRepresentation<PointNormalT>::nr_dimensions_;
public:
	MyPointRepresentation()
	{
		// Define the number of dimensions
		nr_dimensions_ = 4;
	}

	// Override the copyToFloatArray method to define our feature vector
	virtual void copyToFloatArray(const PointNormalT &p, float * out) const
	{
		// < x, y, z, curvature >
		out[0] = p.x;
		out[1] = p.y;
		out[2] = p.z;
		out[3] = p.curvature;
	}
};

#if defined(_MSC_VER) && _MSC_VER >= 1900
#pragma warning(pop) 
#endif 