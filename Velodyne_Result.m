% % The GNU General Public License
% % 
% % Copyright (C) 2017 Yung Feng Chang
% % 
% % This program is free software: you can redistribute it and/or modify
% % it under the terms of the GNU General Public License as published by
% % the Free Software Foundation, either version 3 of the License, or
% % (at your option) any later version.
% % 
% % This program is distributed in the hope that it will be useful,
% % but WITHOUT ANY WARRANTY; without even the implied warranty of
% % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% % GNU General Public License for more details.
% % 
% % You should have received a copy of the GNU General Public License
% % along with this program.  If not, see <http://www.gnu.org/licenses/>.
%%
clear all ; clc ; close all;

% Read GroundTruth
filename = 'OutPut_H_GroundTruth_00.txt';
GroundTruth = dlmread(filename);
[len_gt, ~] = size(GroundTruth);
H_GroundTruth = GroundTruth(1:4,1:4);

% Initialize 
len_input_gt = len_gt/4 - 400; % frame length
T_x = zeros(len_input_gt,1);
T_y = zeros(len_input_gt,1);
T_z = zeros(len_input_gt,1);

T_x(1,1) = H_GroundTruth(1,4);
T_y(1,1) = H_GroundTruth(2,4);
T_z(1,1) = H_GroundTruth(3,4);

theta_x_gt = zeros(len_input_gt,1);
theta_y_gt = zeros(len_input_gt,1);
theta_z_gt = zeros(len_input_gt,1);

% Get Translation and Rotation from GroundTruth
for i = 1:len_input_gt 
    H_GroundTruth = GroundTruth(4*i + 1:4*i+4,1:4);
    % Translation for GroundTruth
    T_x(i,1) = H_GroundTruth(1,4); 
    T_y(i,1) = H_GroundTruth(3,4); 
    % Rotation for GroundTruth
    R_Sum_gt = H_GroundTruth(1:3,1:3);
    % Rotation Angle Calculation from Rotation Matrix
    theta_x_gt(i,1) = atan2(R_Sum_gt(3,2),R_Sum_gt(3,3));
    theta_y_gt(i,1) = atan2(-R_Sum_gt(3,1),sqrt(R_Sum_gt(3,2)^2 + R_Sum_gt(3,3)^2));
    theta_z_gt(i,1) = atan2(R_Sum_gt(2,1),R_Sum_gt(1,1));
end
%%
% Read Output data from Lidar Odometry
filename = 'OutPut_H_Sequence_00.txt';
M = dlmread(filename);

% Initialize 
[len_m, ~] = size(M);
len_input = len_m/4 - 400; 
H_Sum = M(1:4,1:4);
T_x1 = zeros(len_input,1);
T_y1 = zeros(len_input,1);
T_z1 = zeros(len_input,1);
theta_x = zeros(len_input,1);
theta_y = zeros(len_input,1);
theta_z = zeros(len_input,1);

T_x1(1,1) = H_Sum(1,4);
T_y1(1,1) = H_Sum(2,4);
T_z1(1,1) = H_Sum(3,4);

degree = pi / 2;
R = [cos(degree) -sin(degree);sin(degree) cos(degree)];

% Get Translation and Rotation from Lidar Odometry
for i = 1:len_input - 1
    H_Sum = H_Sum*M(4*i + 1:4*i+4,1:4);
    H_Sum_new = R*[H_Sum(1,4);H_Sum(2,4)];
    % Translation for Lidar Odometry
    T_x1(i,1) = H_Sum_new(1,1); 
    T_y1(i,1) = H_Sum_new(2,1); 
    % Rotation for Lidar Odometry
    R_Sum = H_Sum(1:3,1:3);
    % Rotation Angle for Lidar Odometry
    theta_x(i,1) = atan2(R_Sum(3,2),R_Sum(3,3));
    theta_y(i,1) = atan2(-R_Sum(3,1),sqrt(R_Sum(3,2)^2 + R_Sum(3,3)^2));
    theta_z(i,1) = atan2(R_Sum(2,1),R_Sum(1,1));
end

%Plot Path 
figure,
plot(T_x(1:len_input_gt - 1),T_y(1:len_input_gt - 1),'r-')
hold on
plot(T_x1(1:len_input - 1),-T_y1(1:len_input - 1),'b-') 
hold on
plot(T_x1(1,1),-T_y1(1,1),'gs')
hold on

legend('Ground Truth','Lidar Odometry','origin') %,
xlabel('x[m]')
ylabel('z[m]')
axis equal

% Calculate Translation Error vs Path Length
T_x_diff = T_x - T_x1(1:end-1);
T_y_diff = T_y + T_y1(1:end-1);

T_error = sqrt(T_x_diff.^2 + T_y_diff.^2).*100 ./ (sqrt(T_x.^2 + T_y.^2)); 
T_error_x = linspace(100,298,199); 
T_error_y = T_error(100:298,1); 
figure,
subplot(2,1,1);
plot(T_error_x,T_error_y,'sb-')
legend('Translation Error')
xlabel('Path Length [m]')
ylabel('Translation Error [%]')

% Calculate Rotation Error vs Path Length
theta_x_diff = theta_x_gt - theta_x(1:end-1);
theta_y_diff = theta_y_gt - theta_y(1:end-1);
theta_z_diff = theta_z_gt - theta_z(1:end-1);
R_error = sqrt(theta_x_diff.^2 + theta_y_diff.^2 + theta_z_diff.^2) ./ (1000.*sqrt(theta_x_gt.^2 + theta_y_gt.^2 + theta_z_gt.^2)); 
R_error_x = linspace(100,298,199); 
R_error_y = R_error(100:298,1); 

subplot(2,1,2);
plot(R_error_x,R_error_y,'sb-')
legend('Rotation Error')
xlabel('Path Length [m]')
ylabel('Rotation Error [deg/m]')