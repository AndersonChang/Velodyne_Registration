The aim of this project is to apply Lidar Registration for Velodyne KITTI dataset. 

The "Velodyne.cpp" will apply voxel grid filter to downsampling input point cloud, then compute normal vectors of point cloud.  For the next step, use nonlinear ICP to keep iterating until the distance between
two clouds less than threshold. 

The "Velodyne_Result.m" will plot the final result, translation error rate and rotation error rate.

How to use:

1) Download velodyne kitti dataset and paste the path to sprintf(filename, "your path\\%06d.bin", numFrame)
2) Set MAX_FRAME to the corresponding frames in your dataset.
3) Run Velodyne.cpp, it will generate "OutPut_H" to save all homogeneous matrices.
4) Run Velodyne_Result.m to plot the result.

Datasets can be downloaded at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php