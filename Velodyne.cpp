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

#define MAX_FRAME 300

int cloudSortInd[1200];
int cloudNeighborPicked[1200];

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

	//p->spin();
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
	for (int i = 0; i < 30; ++i)
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

		int cloudSize = 0;
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
		//p = new pcl::visualization::PCLVisualizer("Pairwise Incremental Registration example");
		//p->createViewPort(0.0, 0, 0.5, 1.0, vp_1);
		//p->createViewPort(0.5, 0, 1.0, 1.0, vp_2);
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

		pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
		icp.setInputCloud(cloud_in_filtered);
		icp.setInputTarget(cloud_out_filtered);
		pcl::PointCloud<pcl::PointXYZ> Final;
		icp.setMaximumIterations(50);
		//icp.setEuclideanFitnessEpsilon(5);
		icp.align(Final);
		//std::cout << "has converged:" << icp.hasConverged() << " score: " <<
		//	icp.getFitnessScore() << std::endl;
		//std::cout << icp.getFinalTransformation() << std::endl;
		Eigen::Matrix4f homogenous_matrix;
		homogenous_matrix = icp.getFinalTransformation();

		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				R.at<float>(i, j) = homogenous_matrix(i, j);
			}
		}

		for (int i = 0; i < 3; i++)
		{
			t.at<float>(i, 0) = homogenous_matrix(i, 3);
		}

		t_f = t_f + R_f*t;
		R_f = R*R_f;
		Final_t.push_back({t_f.at<float>(0),t_f.at<float>(2)});
		Final_H.push_back(homogenous_matrix);
	}

	//for (int i = 0; i < Final_t.size(); i++)
	//{
	//	for (int j = 0; j < Final_t[0].size(); j++)
	//	{
	//		cout << Final_t[i][j] << " ";
	//	}
	//	cout << endl;
	//}
	
	MyFileOutput(Final_t);
	MyFileOutput_Mat(Final_H);

	//namedWindow("Trajectory", WINDOW_AUTOSIZE);// Create a window for display.
	//resizeWindow("Trajectory", 600, 600);
	//for (int i = 0; i < Final_t.size(); i++)
	//{
	//	//cout << Final_t[i][0] << endl;
	//	//cout << Final_t[i][1] << endl;
	//	int x = int(Final_t[i][0]) + 450;
	//	int y = int(Final_t[i][1]) + 300;

	//	rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
	//	sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<float>(0), t_f.at<float>(1), t_f.at<float>(2));
	//	putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 1);
	//	circle(traj, Point(x, y), 2, CV_RGB(255, 0, 0), 2);
	//	imshow("Trajectory", traj);
	//	waitKey(1);
	//}

	auto t2 = Clock::now();
	cout << "time for nms: " << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() / (double)1000000
		<< " seconds" << endl;
	system("pause");
	return 0;
}

#if defined(_MSC_VER) && _MSC_VER >= 1900
#pragma warning(pop) 
#endif 


//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
//PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
//if (pcl::io::loadPLYFile<pcl::PointXYZRGB>("final_scene.ply", *cloud) == -1) //* load the file
//{
//	PCL_ERROR("Couldn't read file test_pcd.pcd \n");
//	return (-1);
//}

//viewer->setBackgroundColor(0, 0, 0);
////pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color(cloud, 255, 0, 0);
//viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "sample cloud");
//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
//viewer->addCoordinateSystem(1.0);
//viewer->initCameraParameters();


//while (!viewer->wasStopped())
//{
//	viewer->spinOnce(100);
//	boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//}