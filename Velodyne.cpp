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

#include "Velodyne.h"

int main(int argc, char** argv)
{
	// Timer
	auto t1 = Clock::now();
	char filename[200];
	vector<PointCloud<PointXYZI>::Ptr> frames;
	vector<Eigen::Matrix4f> Final_H;

    // Read .bin files and save point cloud objects to a vector
	for (int numFrame = 0; numFrame < MAX_FRAME; numFrame++)
	{
		sprintf(filename, "D:\\Downloads\\data_odometry_velodyne\\dataset\\sequences\\00\\velodyne\\%06d.bin", numFrame);
		PointCloud<PointXYZI>::Ptr frame_points = MyFileOpen(filename);
		frames.push_back(frame_points);
	}

	// Lidar Registration from 0 to Max_Frame.
	for (int numFrame = 0; numFrame < MAX_FRAME - 1; numFrame++)
	{

/************************************************************************************************/
		// File IO
		PointCloud<PointXYZ>::Ptr cloud_in(new PointCloud<PointXYZ>);
		PointCloud<PointXYZ>::Ptr cloud_out(new PointCloud<PointXYZ>);

		// Fill in CloudIn data from i frame
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

		// Fill in Cloudout data from i + 1 frame
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

/************************************************************************************************/
		// Apply Voxel Grid Filter to downsampling the data.
		PointCloud<PointXYZ>::Ptr cloud_in_filtered(new PointCloud<PointXYZ>);
		PointCloud<PointXYZ>::Ptr cloud_out_filtered(new PointCloud<PointXYZ>);

		// Create the filtering object
		VoxelGrid<PointXYZ> sor;

		sor.setInputCloud(cloud_in);
		sor.setLeafSize(0.8f, 0.8f, 0.8f);
		sor.filter(*cloud_in_filtered);

		sor.setInputCloud(cloud_out);
		sor.setLeafSize(0.8f, 0.8f, 0.8f);
		sor.filter(*cloud_out_filtered);

/************************************************************************************************/
		// Compute surface normal vectors and curvature features
		PointCloudWithNormals::Ptr points_with_normals_src(new PointCloudWithNormals);
		PointCloudWithNormals::Ptr points_with_normals_tgt(new PointCloudWithNormals);
		NormalEstimation<PointXYZ, PointNormalT> norm_est;

		// KD Tree Structure to search nearest neighbors
		search::KdTree<PointXYZ>::Ptr tree(new search::KdTree<PointXYZ>());
		norm_est.setSearchMethod(tree);
		norm_est.setKSearch(30);

		// Save cloud_in normal vectors 
		norm_est.setInputCloud(cloud_in_filtered);
		norm_est.compute(*points_with_normals_src);
		copyPointCloud(*cloud_in_filtered, *points_with_normals_src);

		// Save cloud_out normal vectors 
		norm_est.setInputCloud(cloud_out_filtered);
		norm_est.compute(*points_with_normals_tgt);
		copyPointCloud(*cloud_out_filtered, *points_with_normals_tgt);

		MyPointRepresentation point_representation;
		float alpha[4] = { 1.0, 1.0, 1.0, 1.0 };
		point_representation.setRescaleValues(alpha);

/************************************************************************************************/
		// Apply non-linear ICP to align two point clouds
		IterativeClosestPointNonLinear<PointNormalT, PointNormalT> nonlinear_icp;
		// The maximum distance between two corresponding points
		nonlinear_icp.setMaxCorrespondenceDistance(100); 
		// The maximum tolerance for transformation increment
		nonlinear_icp.setTransformationEpsilon(0.5);
		// Input normal vectors feature for calculation
		nonlinear_icp.setPointRepresentation(boost::make_shared<const MyPointRepresentation>(point_representation));
		nonlinear_icp.setInputSource(points_with_normals_src);
		nonlinear_icp.setInputTarget(points_with_normals_tgt);

		Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity(), prev, targetToSource;
		PointCloudWithNormals::Ptr nonlinear_icp_result = points_with_normals_src;

		// Keep iterating to let two clouds get closer until the error less than tolerance
		nonlinear_icp.setMaximumIterations(10); 
		for (int i = 0; i < 50; ++i) 
		{
			points_with_normals_src = nonlinear_icp_result;
			nonlinear_icp.setInputSource(points_with_normals_src);
			nonlinear_icp.align(*nonlinear_icp_result);

			// Accumulate transformation between each Iteration
			Ti = nonlinear_icp.getFinalTransformation() * Ti;

			// If the difference between this transformation and the previous one
			// is smaller than the threshold, refine the process by reducing
			// the maximal correspondence distance
			if (fabs((nonlinear_icp.getLastIncrementalTransformation() - prev).sum()) < nonlinear_icp.getTransformationEpsilon())
				nonlinear_icp.setMaxCorrespondenceDistance(nonlinear_icp.getMaxCorrespondenceDistance() - 1); 

			prev = nonlinear_icp.getLastIncrementalTransformation();
		}

/************************************************************************************************/
		// Save Homogeneous matrix to vector container
		Eigen::Matrix4f homogenous_matrix;
		homogenous_matrix = Ti;
		Final_H.push_back(homogenous_matrix);

		// Visualize data
		//p = new pcl::visualization::PCLVisualizer("Pairwise Incremental Registration example");
		//p->createViewPort(0.0, 0, 0.5, 1.0, vp_1);
		//p->createViewPort(0.5, 0, 1.0, 1.0, vp_2);

		//Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity(), pairTransform;
		//PointCloud<PointXYZ>::Ptr output(new PointCloud<PointXYZ>);
		//pcl::transformPointCloud(*cloud_out_filtered, *output, targetToSource);
		//pcl::transformPointCloud(*cloud_in_filtered, *output, targetToSource);

		//p->removePointCloud("source");
		//p->removePointCloud("target");

		//PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_tgt_h(output, 0, 255, 0);
		//PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_src_h(cloud_in_filtered, 255, 0, 0);
		//p->addPointCloud(output, cloud_tgt_h, "target", vp_2);
		//p->addPointCloud(cloud_out_filtered, cloud_src_h, "source", vp_2);
		//p->spin();
	}

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

