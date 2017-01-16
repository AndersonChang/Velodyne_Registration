clear all ; clc ; close all;

% Read Output Data from Visual Odometry
filename = 'OutPut_T_GroundTruth.txt'; %OutPut_T_LinearICP.txt
fid = fopen(filename,'rt');
tmp = textscan(fid,'%s','Delimiter',' ');
fclose(fid);

[len,~] = size(tmp{1,1});
T_x_gt = zeros(len/2 + 1,1);
T_z_gt = zeros(len/2 + 1,1);

m = 1;
n = 1;

for i = 1:len
   if mod(i,2) == 1 
       T_x_gt(m,1) =  str2num(tmp{1,1}{i,1});
       m = m + 1;
   else
       T_z_gt(n,1) =  str2num(tmp{1,1}{i,1});
       n = n + 1;
   end
end

% Read Output data from Visual Odometry
filename = 'Output_H_NonLinearICP_300.txt';
M = dlmread(filename);

[len_m, ~] = size(M);
H_Sum = M(1:4,1:4);

T_x1 = zeros(len_m/4,1);
T_y1 = zeros(len_m/4,1);
T_z1 = zeros(len_m/4,1);

T_x1(1,1) = H_Sum(1,4);
T_y1(1,1) = H_Sum(2,4);
T_z1(1,1) = H_Sum(3,4);

degree = pi/2;
R = [cos(degree) -sin(degree);sin(degree) cos(degree)];

for i = 1:len_m/4 - 1
    H_Sum = H_Sum*M(4*i + 1:4*i+4,1:4);
    H_Sum_new = R*[H_Sum(1,4);H_Sum(2,4)];
    T_x1(i,1) = H_Sum_new(1,1); 
    T_y1(i,1) = H_Sum_new(2,1); 
end

% Plot Path 
figure,
plot(T_x_gt(1:end - 1),T_z_gt(1:end - 1),'r-')
hold on
plot(T_x1(1:end - 1),-T_y1(1:end - 1),'b-') 
hold on
plot(T_x_gt(1,1),T_z_gt(1,1),'gs')
hold on

legend('Ground Truth','Lidar Odometry','origin')
xlabel('x[m]')
ylabel('z[m]')

% Calculate Translation Error vs Path Length
T_x_diff = T_x_gt - T_x1;
T_y_diff = T_z_gt + T_y1;

T_error = abs(T_x_diff + T_y_diff).*100 ./ (sqrt(T_x_gt.^2 + T_z_gt.^2)); 
A = linspace(100,299,200);
B = T_error(100:299,1);
figure,
subplot(2,2,1);
plot(A,B,'sb-')

legend('Translation Error')
xlabel('Path Length[m]')
ylabel('Translation Error[%]')

% Calculate Rotation Error vs Path Length
H_Sum = M(1:4,1:4);
theta_x = zeros(len_m/4,1);
theta_y = zeros(len_m/4,1);
theta_z = zeros(len_m/4,1);

for i = 1:len_m/4 - 1
    H_Sum = H_Sum*M(4*i + 1:4*i+4,1:4);
    R_Sum = H_Sum(1:3,1:3);
    theta_x(i,1) = atan2(R_Sum(3,2),R_Sum(3,3));
    theta_y(i,1) = atan2(-R_Sum(3,1),sqrt(R_Sum(3,2)^2 + R_Sum(3,3)^2));
    theta_z(i,1) = atan2(R_Sum(2,1),R_Sum(1,1));
end


subplot(2,2,2);
legend('Rotation Error')
xlabel('Path Length[m]')
ylabel('Rotation Error[deg/m]')
% Calculate Tranlation Error vs Speed
subplot(2,2,3);
legend('Translation Error')
xlabel('Speed[km/h]')
ylabel('Translation Error[%]')
% Calculate Rotation Error vs Speed

subplot(2,2,4);
legend('Rotation Error')
xlabel('Speed[km/h]')
ylabel('Rotation Error[deg/m]')