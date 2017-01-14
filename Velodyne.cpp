#if defined(_MSC_VER) && _MSC_VER >= 1900
#pragma warning(push) 
#pragma warning(disable:4996) 
#endif 

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector> 
#include <chrono> 

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

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

using namespace cv;
using namespace std;
using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;
using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

#define MAX_FRAME 10 //250

//convenient typedefs
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;
typedef chrono::high_resolution_clock Clock; //timer

// This is a tutorial so we can afford having global variables 
//our visualizer
pcl::visualization::PCLVisualizer *p;
//its left and right viewports
int vp_1, vp_2;

pcl::PointCloud<PointXYZI>::Ptr MyFileOpen(const string& filename)
{

	// load point cloud
	ifstream input(filename, ios::in | ios::binary);
	if (!input.good()) {
		exit(EXIT_FAILURE);
	}
	input.seekg(0, ios::beg);

	pcl::PointCloud<PointXYZI>::Ptr points(new pcl::PointCloud<PointXYZI>);

	int i;
	for (i = 0; input.good() && !input.eof(); i++) {
		PointXYZI point;
		input.read((char *)&point.x, 3 * sizeof(float));
		input.read((char *)&point.intensity, sizeof(float));
		points->push_back(point);
	}
	input.close();

	//for (int i = 0; i < points->size(); i++)
	//{
	//	PointXYZI temp = points->at(1);
	//	cout << temp.x << endl;
	//	cout << temp.y << endl;
	//	cout << temp.z << endl;
	//}

	return points;
}

//convenient structure to handle our pointclouds
struct PCD
{
	PointCloud<PointXYZ>::Ptr cloud;
	std::string f_name;

	PCD() : cloud(new PointCloud<PointXYZ>) {};
};

struct PCDComparator
{
	bool operator () (const PCD& p1, const PCD& p2)
	{
		return (p1.f_name < p2.f_name);
	}
};


// Define a new point representation for < x, y, z, curvature >
class MyPointRepresentation : public pcl::PointRepresentation <PointNormalT>
{
	using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;
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

////////////////////////////////////////////////////////////////////////////////
/** \brief Display source and target on the first viewport of the visualizer
*
*/
void showCloudsLeft(const PointCloud<PointXYZ>::Ptr cloud_target, const PointCloud<PointXYZ>::Ptr cloud_source)
{
	p->removePointCloud("vp1_target");
	p->removePointCloud("vp1_source");

	PointCloudColorHandlerCustom<PointXYZ> tgt_h(cloud_target, 0, 255, 0);
	PointCloudColorHandlerCustom<PointXYZ> src_h(cloud_source, 255, 0, 0);
	p->addPointCloud(cloud_target, tgt_h, "vp1_target", vp_1);
	p->addPointCloud(cloud_source, src_h, "vp1_source", vp_1);

	//for(int i = 1; i < cloud_source->points.size(); i++)
	//	p->addLine<pcl::PointXYZ>(cloud_source->points[i], cloud_source->points[i - 1], "line");
	//p->addSphere(cloud_source->points[0], 0.2, 0.5, 0.5, 0.0, "sphere");

	p->spin();
}


////////////////////////////////////////////////////////////////////////////////
/** \brief Display source and target on the second viewport of the visualizer
*
*/
void showCloudsRight(const PointCloudWithNormals::Ptr cloud_target, const PointCloudWithNormals::Ptr cloud_source)
{
	p->removePointCloud("source");
	p->removePointCloud("target");


	PointCloudColorHandlerGenericField<PointNormalT> tgt_color_handler(cloud_target, "curvature");
	if (!tgt_color_handler.isCapable())
		PCL_WARN("Cannot create curvature color handler!");

	PointCloudColorHandlerGenericField<PointNormalT> src_color_handler(cloud_source, "curvature");
	if (!src_color_handler.isCapable())
		PCL_WARN("Cannot create curvature color handler!");


	p->addPointCloud(cloud_target, tgt_color_handler, "target", vp_2);
	p->addPointCloud(cloud_source, src_color_handler, "source", vp_2);
	p->spinOnce();
}

void pairAlign(const PointCloud<PointXYZ>::Ptr cloud_src, const PointCloud<PointXYZ>::Ptr cloud_tgt, PointCloud<PointXYZ>::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = false)
{
	//
	// Downsample for consistency and speed
	// \note enable this for large datasets
	PointCloud<PointXYZ>::Ptr src(new PointCloud<PointXYZ>);
	PointCloud<PointXYZ>::Ptr tgt(new PointCloud<PointXYZ>);
	pcl::VoxelGrid<PointXYZ> grid;
	if (downsample)
	{
		grid.setLeafSize(0.1f, 0.1f, 0.1f);
		grid.setInputCloud(cloud_src);
		grid.filter(*src);

		grid.setInputCloud(cloud_tgt);
		grid.filter(*tgt);
	}
	else
	{
		src = cloud_src;
		tgt = cloud_tgt;
	}


	// Compute surface normals and curvature
	PointCloudWithNormals::Ptr points_with_normals_src(new PointCloudWithNormals);
	PointCloudWithNormals::Ptr points_with_normals_tgt(new PointCloudWithNormals);

	pcl::NormalEstimation<pcl::PointXYZ, PointNormalT> norm_est;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	norm_est.setSearchMethod(tree);
	norm_est.setKSearch(30);

	norm_est.setInputCloud(src);
	norm_est.compute(*points_with_normals_src);
	pcl::copyPointCloud(*src, *points_with_normals_src);

	norm_est.setInputCloud(tgt);
	norm_est.compute(*points_with_normals_tgt);
	pcl::copyPointCloud(*tgt, *points_with_normals_tgt);

	//
	// Instantiate our custom point representation (defined above) ...
	MyPointRepresentation point_representation;
	// ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
	float alpha[4] = { 1.0, 1.0, 1.0, 1.0 };
	point_representation.setRescaleValues(alpha);

	//
	// Align
	pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
	reg.setTransformationEpsilon(1e-6);
	// Set the maximum distance between two correspondences (src<->tgt) to 10cm
	// Note: adjust this based on the size of your datasets
	reg.setMaxCorrespondenceDistance(0.1);
	// Set the point representation
	reg.setPointRepresentation(boost::make_shared<const MyPointRepresentation>(point_representation));

	reg.setInputSource(points_with_normals_src);
	reg.setInputTarget(points_with_normals_tgt);

	//
	// Run the same optimization in a loop and visualize the results
	Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity(), prev, targetToSource;
	PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
	reg.setMaximumIterations(2);
	for (int i = 0; i < 1; ++i) //30
	{
		// save cloud for visualization purpose
		points_with_normals_src = reg_result;

		// Estimate
		reg.setInputSource(points_with_normals_src);
		reg.align(*reg_result);

		//accumulate transformation between each Iteration
		Ti = reg.getFinalTransformation() * Ti;

		//if the difference between this transformation and the previous one
		//is smaller than the threshold, refine the process by reducing
		//the maximal correspondence distance
		if (fabs((reg.getLastIncrementalTransformation() - prev).sum()) < reg.getTransformationEpsilon())
			reg.setMaxCorrespondenceDistance(reg.getMaxCorrespondenceDistance() - 0.001);

		prev = reg.getLastIncrementalTransformation();

		// visualize current state
		showCloudsRight(points_with_normals_tgt, points_with_normals_src);
	}

	//
	// Get the transformation from target to source
	targetToSource = Ti.inverse();

	//
	// Transform target back in source frame
	pcl::transformPointCloud(*cloud_tgt, *output, targetToSource);

	p->removePointCloud("source");
	p->removePointCloud("target");

	PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_tgt_h(output, 0, 255, 0);
	PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_src_h(cloud_src, 255, 0, 0);
	p->addPointCloud(output, cloud_tgt_h, "target", vp_2);
	p->addPointCloud(cloud_src, cloud_src_h, "source", vp_2);
  	p->spin();

	p->removePointCloud("source");
	p->removePointCloud("target");

	//add the source to the transformed target
	*output += *cloud_src;

	final_transform = targetToSource;
}

void MyFileOutput(vector<vector<float>>& output)
{
	ofstream OutPut_File; //Create a ofstream for csv file output.
	OutPut_File.open("OutPut_T.txt");
	for (auto& row : output)
	{
		for (auto col : row)
			OutPut_File << col << ' ';
		OutPut_File << '\n';
	}
	OutPut_File.close();
}

void MyFileOutput_Mat(vector<Eigen::Matrix4f>& output)
{
	ofstream OutPut_File; //Create a ofstream for csv file output.
	OutPut_File.open("OutPut_H.txt");
	for (auto& row : output)
	{
		/*for (auto col : row)*/
		OutPut_File << row ;
		OutPut_File << '\n';
	}
	OutPut_File.close();
}

int main()
{
	auto t1 = Clock::now();
	char filename[200];
	vector<pcl::PointCloud<PointXYZI>::Ptr> frames;
	vector<Eigen::Matrix4f> Final_H;
	vector<vector<float>> Final_t;

	Mat t_f = Mat::zeros(3, 1, CV_32F);
	Mat R_f = Mat::zeros(3, 3, CV_32F);
	Mat R = Mat::zeros(3, 3, CV_32F);
	Mat t = Mat::zeros(3, 1, CV_32F);
	Mat traj = Mat::zeros(600, 600, CV_8UC3);
	R_f.at<float>(0, 0) = 1;
	R_f.at<float>(1, 1) = 1;
	R_f.at<float>(2, 2) = 1;

	char text[100];
	int fontFace = FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 1;
	cv::Point textOrg(10, 50);

	for (int numFrame = 0; numFrame < MAX_FRAME; numFrame++)
	{
		sprintf(filename, "D:\\Downloads\\data_odometry_velodyne\\dataset\\sequences\\00\\velodyne\\%06d.bin", numFrame); 
		pcl::PointCloud<PointXYZI>::Ptr frame_points = MyFileOpen(filename);
		frames.push_back(frame_points);
	}


	//p = new pcl::visualization::PCLVisualizer("Pairwise Incremental Registration example");
	//p->createViewPort(0.0, 0, 0.5, 1.0, vp_1);
	//p->createViewPort(0.5, 0, 1.0, 1.0, vp_2);
	PointCloud<PointXYZ>::Ptr result(new PointCloud<PointXYZ>);
	Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity(), pairTransform;


	for (int numFrame = 0; numFrame < MAX_FRAME - 1; numFrame++)
	{

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);

		// Fill in the CloudIn data
		cloud_in->width = frames[numFrame]->size();
		cloud_in->height = 1;
		cloud_in->is_dense = false;
		cloud_in->points.resize(cloud_in->width * cloud_in->height);
		for (size_t i = 0; i < cloud_in->points.size(); ++i)
		{
			PointXYZI temp = frames[numFrame]->at(i);
			cloud_in->points[i].x = temp.x;
			cloud_in->points[i].y = temp.y;
			cloud_in->points[i].z = temp.z;
		}

		// Fill in the Cloudout data
		cloud_out->width = frames[numFrame + 1]->size();
		cloud_out->height = 1;
		cloud_out->is_dense = false;
		cloud_out->points.resize(cloud_out->width * cloud_out->height);
		for (size_t i = 0; i < cloud_out->points.size(); ++i)
		{
			PointXYZI temp = frames[numFrame + 1]->at(i);
			cloud_out->points[i].x = temp.x;
			cloud_out->points[i].y = temp.y;
			cloud_out->points[i].z = temp.z;
		}


		// Add visualization data
		//showCloudsLeft(cloud_in, cloud_out);
		p = new pcl::visualization::PCLVisualizer("Pairwise Incremental Registration example");
		p->createViewPort(0.0, 0, 0.5, 1.0, vp_1);
		p->createViewPort(0.5, 0, 1.0, 1.0, vp_2);
		//PointCloud<PointXYZ>::Ptr temp(new PointCloud<PointXYZ>);
		//pairAlign(cloud_in, cloud_out, temp, pairTransform, true);

		//transform current pair into the global transform
		//pcl::transformPointCloud(*temp, *result, GlobalTransform);

		//update the global transform
		//GlobalTransform = GlobalTransform * pairTransform;


		//cout <<"cloud_in: " << cloud_in->width*cloud_in->height << endl;
		//cout <<"cloud_out: "<< cloud_out->width*cloud_out->height << endl;

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		// Create the filtering object
		pcl::VoxelGrid<pcl::PointXYZ> sor;
		sor.setInputCloud(cloud_in);
		sor.setLeafSize(0.1f, 0.1f, 0.1f);
		sor.filter(*cloud_in_filtered);

		sor.setInputCloud(cloud_out);
		sor.setLeafSize(0.1f, 0.1f, 0.1f);
		sor.filter(*cloud_out_filtered);


		// Corner Detection
		vector<int> cloudSortInd;
		vector<int> cloudNeighborPicked;
		vector<float> Sum_DiffBetweenTenNeighbors;
		vector<int> record;
		cloudSortInd.resize(cloud_in_filtered->width * cloud_in_filtered->height);
		cloudNeighborPicked.resize(cloud_in_filtered->width * cloud_in_filtered->height, 0);
		record.resize(cloud_in_filtered->width * cloud_in_filtered->height, 0);
		Sum_DiffBetweenTenNeighbors.resize(cloud_in_filtered->width * cloud_in_filtered->height);

		for (size_t i = 0; i < cloud_in_filtered->points.size(); ++i)
		{
			//PointXYZI temp = frames[numFrame]->at(i);
			//cloud_in_filtered->points[i].x = temp.x;
			//cloud_in_filtered->points[i].y = temp.y;
			//cloud_in_filtered->points[i].z = temp.z;
			cloudSortInd[i] = i;
		}

		for (int i = 5; i < cloud_in_filtered->points.size() - 5; i++) {
			float diffX = cloud_in_filtered->points[i - 5].x + cloud_in_filtered->points[i - 4].x
				+ cloud_in_filtered->points[i - 3].x + cloud_in_filtered->points[i - 2].x
				+ cloud_in_filtered->points[i - 1].x - 10 * cloud_in_filtered->points[i].x
				+ cloud_in_filtered->points[i + 1].x + cloud_in_filtered->points[i + 2].x
				+ cloud_in_filtered->points[i + 3].x + cloud_in_filtered->points[i + 4].x
				+ cloud_in_filtered->points[i + 5].x;
			float diffY = cloud_in_filtered->points[i - 5].y + cloud_in_filtered->points[i - 4].y
				+ cloud_in_filtered->points[i - 3].y + cloud_in_filtered->points[i - 2].y
				+ cloud_in_filtered->points[i - 1].y - 10 * cloud_in_filtered->points[i].y
				+ cloud_in_filtered->points[i + 1].y + cloud_in_filtered->points[i + 2].y
				+ cloud_in_filtered->points[i + 3].y + cloud_in_filtered->points[i + 4].y
				+ cloud_in_filtered->points[i + 5].y;
			float diffZ = cloud_in_filtered->points[i - 5].z + cloud_in_filtered->points[i - 4].z
				+ cloud_in_filtered->points[i - 3].z + cloud_in_filtered->points[i - 2].z
				+ cloud_in_filtered->points[i - 1].z - 10 * cloud_in_filtered->points[i].z
				+ cloud_in_filtered->points[i + 1].z + cloud_in_filtered->points[i + 2].z
				+ cloud_in_filtered->points[i + 3].z + cloud_in_filtered->points[i + 4].z
				+ cloud_in_filtered->points[i + 5].z;

			Sum_DiffBetweenTenNeighbors[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
		}

		for (int i = 5; i < cloud_in_filtered->points.size() - 6; i++) {
			float diffX = cloud_in_filtered->points[i + 1].x - cloud_in_filtered->points[i].x;
			float diffY = cloud_in_filtered->points[i + 1].y - cloud_in_filtered->points[i].y;
			float diffZ = cloud_in_filtered->points[i + 1].z - cloud_in_filtered->points[i].z;
			float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;

			if (diff > 50) { //0.05

				float depth1 = sqrt(cloud_in_filtered->points[i].x * cloud_in_filtered->points[i].x +
					cloud_in_filtered->points[i].y * cloud_in_filtered->points[i].y +
					cloud_in_filtered->points[i].z * cloud_in_filtered->points[i].z);

				float depth2 = sqrt(cloud_in_filtered->points[i + 1].x * cloud_in_filtered->points[i + 1].x +
					cloud_in_filtered->points[i + 1].y * cloud_in_filtered->points[i + 1].y +
					cloud_in_filtered->points[i + 1].z * cloud_in_filtered->points[i + 1].z);

				if (depth1 > depth2) {
					diffX = cloud_in_filtered->points[i + 1].x - cloud_in_filtered->points[i].x * depth2 / depth1;
					diffY = cloud_in_filtered->points[i + 1].y - cloud_in_filtered->points[i].y * depth2 / depth1;
					diffZ = cloud_in_filtered->points[i + 1].z - cloud_in_filtered->points[i].z * depth2 / depth1;

					if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth2 < 0.1) {
						cloudNeighborPicked[i - 5] = 1;
						cloudNeighborPicked[i - 4] = 1;
						cloudNeighborPicked[i - 3] = 1;
						cloudNeighborPicked[i - 2] = 1;
						cloudNeighborPicked[i - 1] = 1;
						cloudNeighborPicked[i] = 1;
					}
				}
				else {
					diffX = cloud_in_filtered->points[i + 1].x * depth1 / depth2 - cloud_in_filtered->points[i].x;
					diffY = cloud_in_filtered->points[i + 1].y * depth1 / depth2 - cloud_in_filtered->points[i].y;
					diffZ = cloud_in_filtered->points[i + 1].z * depth1 / depth2 - cloud_in_filtered->points[i].z;

					if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth1 < 0.1) {
						cloudNeighborPicked[i + 1] = 1;
						cloudNeighborPicked[i + 2] = 1;
						cloudNeighborPicked[i + 3] = 1;
						cloudNeighborPicked[i + 4] = 1;
						cloudNeighborPicked[i + 5] = 1;
						cloudNeighborPicked[i + 6] = 1;
					}
				}
			}

			float diffX2 = cloud_in_filtered->points[i].x - cloud_in_filtered->points[i - 1].x;
			float diffY2 = cloud_in_filtered->points[i].y - cloud_in_filtered->points[i - 1].y;
			float diffZ2 = cloud_in_filtered->points[i].z - cloud_in_filtered->points[i - 1].z;
			float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;

			float dis = cloud_in_filtered->points[i].x * cloud_in_filtered->points[i].x
				+ cloud_in_filtered->points[i].y * cloud_in_filtered->points[i].y
				+ cloud_in_filtered->points[i].z * cloud_in_filtered->points[i].z;

			if (diff > (0.25 * 0.25) / (20 * 20) * dis && diff2 > (0.25 * 0.25) / (20 * 20) * dis) {
				cloudNeighborPicked[i] = 1;
			}
		}

		int count_one = count(cloudNeighborPicked.begin(), cloudNeighborPicked.end(), 1);
		cout << count_one << endl;

		pcl::PointCloud<pcl::PointXYZ>::Ptr cornerPointsSharp(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::PointCloud<pcl::PointXYZ>::Ptr cornerPointsLessSharp(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::PointCloud<pcl::PointXYZ>::Ptr surfPointsFlat(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::PointCloud<pcl::PointXYZ>::Ptr surfPointsLessFlat(new pcl::PointCloud<pcl::PointXYZ>());

		int startPoints[4] = { 5, 6 + int((cloud_in_filtered->points.size() - 10) / 4.0),
			6 + int((cloud_in_filtered->points.size() - 10) / 2.0), 6 + int(3 * (cloud_in_filtered->points.size() - 10) / 4.0) };
		int endPoints[4] = { 5 + int((cloud_in_filtered->points.size() - 10) / 4.0), 5 + int((cloud_in_filtered->points.size() - 10) / 2.0),
			5 + int(3 * (cloud_in_filtered->points.size() - 10) / 4.0), cloud_in_filtered->points.size() - 6 };

		for (int i = 0; i < 4; i++) {
			int sp = startPoints[i];
			int ep = endPoints[i];

			for (int j = sp + 1; j <= ep; j++) {
				for (int k = j; k >= sp + 1; k--) {
					if (Sum_DiffBetweenTenNeighbors[cloudSortInd[k]] < Sum_DiffBetweenTenNeighbors[cloudSortInd[k - 1]]) {
						int temp = cloudSortInd[k - 1];
						cloudSortInd[k - 1] = cloudSortInd[k];
						cloudSortInd[k] = temp;
					}
				}
			}

			int largestPickedNum = 0;
			for (int j = ep; j >= sp; j--) {

				//cout << fabs(cloud_in_filtered->points[cloudSortInd[j]].x) << endl;
				//cout << fabs(cloud_in_filtered->points[cloudSortInd[j]].y) << endl;
				//cout << fabs(cloud_in_filtered->points[cloudSortInd[j]].z) << endl;

				if (cloudNeighborPicked[cloudSortInd[j]] == 0 &&
					Sum_DiffBetweenTenNeighbors[cloudSortInd[j]] > 0.1 && //0.1 1000
					(fabs(cloud_in_filtered->points[cloudSortInd[j]].x) > 0.3 ||
						fabs(cloud_in_filtered->points[cloudSortInd[j]].y) > 0.3 ||
						fabs(cloud_in_filtered->points[cloudSortInd[j]].z) > 0.3) &&
					fabs(cloud_in_filtered->points[cloudSortInd[j]].x) < 100 && //30
					fabs(cloud_in_filtered->points[cloudSortInd[j]].y) < 100 &&
					fabs(cloud_in_filtered->points[cloudSortInd[j]].z) < 100) {

					largestPickedNum++;
					if (largestPickedNum <= 10000) { //2
						record[cloudSortInd[j]] = 2; 
						cornerPointsSharp->push_back(cloud_in_filtered->points[cloudSortInd[j]]);
					}
					else if (largestPickedNum <= 20000) { //20
						record[cloudSortInd[j]] = 1;
						cornerPointsLessSharp->push_back(cloud_in_filtered->points[cloudSortInd[j]]);
					}
					else {
						break;
					}

					cloudNeighborPicked[cloudSortInd[j]] = 1;
					for (int k = 1; k <= 5; k++) {
						float diffX = cloud_in_filtered->points[cloudSortInd[j] + k].x
							- cloud_in_filtered->points[cloudSortInd[j] + k - 1].x;
						float diffY = cloud_in_filtered->points[cloudSortInd[j] + k].y
							- cloud_in_filtered->points[cloudSortInd[j] + k - 1].y;
						float diffZ = cloud_in_filtered->points[cloudSortInd[j] + k].z
							- cloud_in_filtered->points[cloudSortInd[j] + k - 1].z;
						if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
							break;
						}

						cloudNeighborPicked[cloudSortInd[j] + k] = 1;
					}
					for (int k = -1; k >= -5; k--) {
						float diffX = cloud_in_filtered->points[cloudSortInd[j] + k].x
							- cloud_in_filtered->points[cloudSortInd[j] + k + 1].x;
						float diffY = cloud_in_filtered->points[cloudSortInd[j] + k].y
							- cloud_in_filtered->points[cloudSortInd[j] + k + 1].y;
						float diffZ = cloud_in_filtered->points[cloudSortInd[j] + k].z
							- cloud_in_filtered->points[cloudSortInd[j] + k + 1].z;
						if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
							break;
						}

						cloudNeighborPicked[cloudSortInd[j] + k] = 1;
					}
				}
			}

			int smallestPickedNum = 0;
			for (int j = sp; j <= ep; j++) {
				if (cloudNeighborPicked[cloudSortInd[j]] == 0 &&
					Sum_DiffBetweenTenNeighbors[cloudSortInd[j]] < 0.1 &&
					(fabs(cloud_in_filtered->points[cloudSortInd[j]].x) > 0.3 ||
						fabs(cloud_in_filtered->points[cloudSortInd[j]].y) > 0.3 ||
						fabs(cloud_in_filtered->points[cloudSortInd[j]].z) > 0.3) &&
					fabs(cloud_in_filtered->points[cloudSortInd[j]].x) < 30 &&
					fabs(cloud_in_filtered->points[cloudSortInd[j]].y) < 30 &&
					fabs(cloud_in_filtered->points[cloudSortInd[j]].z) < 30) {

					record[cloudSortInd[j]] = -1;
					surfPointsFlat->push_back(cloud_in_filtered->points[cloudSortInd[j]]);

					smallestPickedNum++;
					if (smallestPickedNum >= 4) {
						break;
					}

					cloudNeighborPicked[cloudSortInd[j]] = 1;
					for (int k = 1; k <= 5; k++) {
						float diffX = cloud_in_filtered->points[cloudSortInd[j] + k].x
							- cloud_in_filtered->points[cloudSortInd[j] + k - 1].x;
						float diffY = cloud_in_filtered->points[cloudSortInd[j] + k].y
							- cloud_in_filtered->points[cloudSortInd[j] + k - 1].y;
						float diffZ = cloud_in_filtered->points[cloudSortInd[j] + k].z
							- cloud_in_filtered->points[cloudSortInd[j] + k - 1].z;
						if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
							break;
						}

						cloudNeighborPicked[cloudSortInd[j] + k] = 1;
					}
					for (int k = -1; k >= -5; k--) {
						float diffX = cloud_in_filtered->points[cloudSortInd[j] + k].x
							- cloud_in_filtered->points[cloudSortInd[j] + k + 1].x;
						float diffY = cloud_in_filtered->points[cloudSortInd[j] + k].y
							- cloud_in_filtered->points[cloudSortInd[j] + k + 1].y;
						float diffZ = cloud_in_filtered->points[cloudSortInd[j] + k].z
							- cloud_in_filtered->points[cloudSortInd[j] + k + 1].z;
						if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
							break;
						}

						cloudNeighborPicked[cloudSortInd[j] + k] = 1;
					}
				}
			}
		}

		for (int i = 0; i < cloud_in_filtered->points.size(); i++) {
			if (record[i] == 0) {
				surfPointsLessFlat->push_back(cloud_in_filtered->points[i]);
			}
		}

		showCloudsLeft(cloud_in_filtered, cornerPointsSharp);
		//PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_tgt_h(cornerPointsSharp, 0, 255, 0);
		//PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_src_h(cloud_in_filtered, 255, 0, 0);
		//p->addPointCloud(cornerPointsSharp, cloud_tgt_h, "target", vp_1);
		//p->addPointCloud(cloud_in_filtered, cloud_src_h, "source", vp_1);
		//p->spin();


		/*Linear ICP*/
		PointCloud<PointXYZ>::Ptr output(new PointCloud<PointXYZ>);
		showCloudsLeft(cloud_in_filtered, cloud_out_filtered);

		pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
		icp.setInputCloud(cloud_in_filtered);
		icp.setInputTarget(cloud_out_filtered);
		pcl::PointCloud<pcl::PointXYZ> Final;
		//icp.setMaximumIterations(10);
		//icp.setEuclideanFitnessEpsilon(0.001);
		icp.align(Final);
		//std::cout << "has converged:" << icp.hasConverged() << " score: " <<
		//	icp.getFitnessScore() << std::endl;
		//std::cout << icp.getFinalTransformation() << std::endl;
		Eigen::Matrix4f homogenous_matrix;
		homogenous_matrix = icp.getFinalTransformation();
		Eigen::Matrix4f targetToSource = homogenous_matrix.inverse();
		//// Visualize
		pcl::transformPointCloud(*cloud_out_filtered, *output, targetToSource);

		p->removePointCloud("source");
		p->removePointCloud("target");

		PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_tgt_h(output, 0, 255, 0);
		PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_src_h(cloud_in_filtered, 255, 0, 0);
		p->addPointCloud(output, cloud_tgt_h, "target", vp_2);
		p->addPointCloud(cloud_in_filtered, cloud_src_h, "source", vp_2);
		p->spin();

		/////////////////////////////////////////////////////////////////////////////////////
		// Compute surface normals and curvature

		////PointCloudWithNormals::Ptr points_with_normals_src(new PointCloudWithNormals);
		////PointCloudWithNormals::Ptr points_with_normals_tgt(new PointCloudWithNormals);

		////pcl::NormalEstimation<pcl::PointXYZ, PointNormalT> norm_est;
		////pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		////norm_est.setSearchMethod(tree);
		////norm_est.setKSearch(30);

		////norm_est.setInputCloud(cloud_in_filtered);
		////norm_est.compute(*points_with_normals_src);
		////pcl::copyPointCloud(*cloud_in_filtered, *points_with_normals_src);

		////norm_est.setInputCloud(cloud_out_filtered);
		////norm_est.compute(*points_with_normals_tgt);
		////pcl::copyPointCloud(*cloud_out_filtered, *points_with_normals_tgt);

		//////
		////// Instantiate our custom point representation (defined above) ...
		////MyPointRepresentation point_representation;
		////// ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
		////float alpha[4] = { 1.0, 1.0, 1.0, 1.0 };
		////point_representation.setRescaleValues(alpha);

		//////
		////// Align
		////pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
		////reg.setTransformationEpsilon(1e-6);
		////// Set the maximum distance between two correspondences (src<->tgt) to 10cm
		////// Note: adjust this based on the size of your datasets
		////reg.setMaxCorrespondenceDistance(0.1);
		////// Set the point representation
		////reg.setPointRepresentation(boost::make_shared<const MyPointRepresentation>(point_representation));

		////reg.setInputSource(points_with_normals_src);
		////reg.setInputTarget(points_with_normals_tgt);

		//////
		////// Run the same optimization in a loop and visualize the results
		////Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity(), prev, targetToSource;
		////PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
		////reg.setMaximumIterations(2);
		////for (int i = 0; i < 1; ++i)
		////{
		////	// save cloud for visualization purpose
		////	points_with_normals_src = reg_result;

		////	// Estimate
		////	reg.setInputSource(points_with_normals_src);
		////	reg.align(*reg_result);

		////	//accumulate transformation between each Iteration
		////	Ti = reg.getFinalTransformation() * Ti;

		////	//if the difference between this transformation and the previous one
		////	//is smaller than the threshold, refine the process by reducing
		////	//the maximal correspondence distance
		////	if (fabs((reg.getLastIncrementalTransformation() - prev).sum()) < reg.getTransformationEpsilon())
		////		reg.setMaxCorrespondenceDistance(reg.getMaxCorrespondenceDistance() - 0.001);

		////	prev = reg.getLastIncrementalTransformation();
		////}
		/////////////////////////////////////////////////////////////////////////////////////////////////////////
		//Eigen::Matrix4f homogenous_matrix;
		//homogenous_matrix = Ti;

		//for (int i = 0; i < 3; i++)
		//{
		//	for (int j = 0; j < 3; j++)
		//	{
		//		R.at<float>(i, j) = homogenous_matrix(i, j);
		//	}
		//}

		//for (int i = 0; i < 3; i++)
		//{
		//	t.at<float>(i, 0) = homogenous_matrix(i, 3);
		//}

		t_f = t_f + R_f*t;
		R_f = R*R_f;
		Final_t.push_back({t_f.at<float>(0),t_f.at<float>(2)});
		Final_H.push_back(homogenous_matrix);
	}

	MyFileOutput(Final_t);
	MyFileOutput_Mat(Final_H);

	auto t2 = Clock::now();
	cout << "time for nms: " << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() / (double)1000000
		<< " seconds" << endl;
	system("pause");
	return 0;
}

#if defined(_MSC_VER) && _MSC_VER >= 1900
#pragma warning(pop) 
#endif 


//cout << "cloud_in_filtered: " << cloud_in_filtered->width*cloud_in_filtered->height << endl;
//cout << "cloud_out_filtered: " << cloud_out_filtered->width*cloud_out_filtered->height << endl;


//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

//viewer->setBackgroundColor(0, 0, 0);
//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud_in_filtered, 0, 255, 0);
//viewer->addPointCloud<pcl::PointXYZ>(cloud_in_filtered, single_color, "sample cloud");
//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color2(cloud_out_filtered, 255, 0, 0);
//viewer->addPointCloud<pcl::PointXYZ>(cloud_out_filtered, single_color2, "sample cloud2");
//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud2");
//viewer->addCoordinateSystem(1.0);
//viewer->initCameraParameters();


//while (!viewer->wasStopped())
//{
//	viewer->spinOnce(100);
//	boost::this_thread::sleep(boost::posix_time::microseconds(1000)); //00
//}