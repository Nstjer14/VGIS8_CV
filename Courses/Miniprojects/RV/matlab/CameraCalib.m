%% Setting up webcam
clear all;
clc;

cam = webcam(1);
%cam.Resolution = '320x240';
%cam.Resolution = '800x600';
%cam.Resolution = '1280x720';
cam.Resolution = '1920x1080';
%cam.Resolution = '2304x1536';
cam.ExposureMode = 'manual';
cam.Exposure = -6;

figure(1);
imshow(cam.snapshot);


%% UR connection Setup
addpath(genpath('ur_interface'));
addpath(genpath('thomas_lib'));
addpath(genpath('calibrationfolder'));


% My static IP is set to 192.168.0.20
% Get the robot ip from the teach pendant: File -> About
robot_ip = '192.168.0.2';
%  robot_ip = '127.0.0.1';         % URsim

sock = tcpip(robot_ip, 30000, 'NetworkRole', 'server');
fclose(sock);
disp('Press Play on robot');
fopen(sock);
disp('Connected!');

homepos = [156.8230 -387.7130  560.1100   -3.1296    0.0074    0.0004];
currentPos = urReadPosC(sock);
if currentPos ~ homepos
urMoveL(sock,homepos);
end
homeimg = snapshot(cam);
figure(2)
imshow(homeimg);
%imwrite(homeimg, ['calibrationfolder/' '3pt_calibration' '.bmp'], 'bmp')
%%
%RP1 = [18.3821 -679.0230  141.8320   -3.1296    0.0074    0.0004];
%RP2 = [-160.4770 -498.0370  141.8320   -3.1296    0.0074    0.0004];
%RP3 = [23.1680 -316.3760  141.8320   -3.1296    0.0074    0.0004];
%figure(1);
%imshow(imgcali);

IP1 = [65 1026];
IP2 = [63 234];
IP3 = [867 234];

RP1 = [18.3821 -679.0230];
RP2 = [-160.4770 -498.0370];
RP3 = [23.1680 -316.3760];

X_rob = [RP1(1) RP2(1) RP3(1)]';
Y_rob = [RP1(2) RP2(2) RP3(2)]';

img_mat = [1 65 1026; 1 63 234; 1 867 234;];
theta = linsolve(img_mat,X_rob); % X coordinate coefficients
phi = linsolve(img_mat,Y_rob); % Y coordinate coefficients

%newP = [799 526]
%x_test = [1 newP]*theta;
%y_test = [1 newP]*phi;


urMoveL(sock, [x_test y_test 144.8210    3.1288    0.0920   -0.0342]);

%%
%    KK = [cameraParams.FocalLength(1)   0    cameraParams.PrincipalPoint(1);
%          0     cameraParams.FocalLength(2)  cameraParams.PrincipalPoint(2);
%          0       0     1];
%    CameraToObjectDistance = cameraParams.TranslationVectors(4,3);
%ToolPose = GetSO4FromURpose(homepos);
%BrickPos = PixelToWorldCoordinate(IP1', ToolPose, KK, CameraToObjectDistance);
          
%%


%%


CalibrationPositions = zeros(10, 6);

CalibrationPositions(1,:) = [139.1550 -393.7310  489.3840    3.0792   -0.0669   -0.0013];
CalibrationPositions(2,:) = [160.3390 -272.0300  534.2310    2.7343    0.3909   -0.0739];
CalibrationPositions(3,:) = [518.4520 -478.9050  365.8380   -2.6896   -0.6276    0.9528]; 
CalibrationPositions(4,:) = homepos;%[-20.5769 -145.9410  568.8740    2.6587    0.2319    0.2597];
CalibrationPositions(5,:) = [54.1598 -231.4180  790.5660    2.9503    0.1298    0.0492];
CalibrationPositions(6,:) = [143.5610 -581.1970  463.9550   -2.8527    0.3738    0.0184];
CalibrationPositions(7,:) = [303.4300 -651.1850  349.2800   -2.6034    0.1377    0.3284];
CalibrationPositions(8,:) = [-86.8422 -443.9080  485.2960   -2.6613    0.9918   -0.7759];
CalibrationPositions(9,:) = [515.5750 -257.0430  550.4940   -2.7543   -1.0659    0.8095];
CalibrationPositions(10,:) =[-64.9401 -299.5080  488.1570    2.8103   -0.7727    0.4979];


%i = 0;

%i = i +1;

for (i = 1:length(CalibrationPositions))   
urMoveL(sock,CalibrationPositions(i,:));
pause(2);
img = snapshot(cam);
imwrite(img, ['calibrationfolder/' num2str(i) '.bmp'], 'bmp')
imshow(img);
end



%%

