clear all;
clc;

cam = webcam(1);
%cam.Resolution = '320x240';
%cam.Resolution = '800x600';
%cam.Resolution = '1280x720';
cam.Resolution = '1920x1080';
%cam.Resolution = '2304x1536';
cam.ExposureMode = 'manual';
cam.Exposure = -3;

figure(1);
imshow(cam.snapshot);

%%
Robot_IP = '192.168.0.2';
robot = tcpip(Robot_IP,30000,'NetworkRole','server');
fclose(robot);
disp('Press Play on Robot...')
fopen(robot);
disp('Connected!');

%%
RobotInitialPose = readrobotpose(robot)
T = RobotInitialPose(1:3); % in mm
O = RobotInitialPose(4:6);

%OpenGripper(robot);

%%
CalibrationPositions = zeros(5, 6);

CalibrationPositions(1,:) = [-403.6970 -414.0400  523.2040    2.4288   -1.7304    0.0047];
CalibrationPositions(2,:) = [-584.9590 -144.2430  467.5790    1.6753   -2.2216    0.7690];
CalibrationPositions(3,:) = [-358.3570 -277.6200  632.8920    2.1100   -1.6195    0.4195];
CalibrationPositions(4,:) = [-235.2380 -506.2140  522.1020    2.3565   -1.1456    0.1063];
CalibrationPositions(5,:) = [ -213.5370 -407.2580  603.7540    2.3002   -1.1276    0.0896];
%CalibrationPositions(6,:) = [520 -330 350 pi -0.45 0.0];
%CalibrationPositions(7,:) = [510 -380 300 pi+0.2 0.0 0.2];
%CalibrationPositions(8,:) = [400 -380 300 pi+0.4 0.4 0.5];
%CalibrationPositions(9,:) = [520 0 300 pi-0.5 0 0.0];
%CalibrationPositions(10,:) = [380 0 300 pi-0.5 0.3 0.4];
%CalibrationPositions(11,:) = [380 100 280 pi-0.5 0.3 0.4];
%CalibrationPositions(12,:) = [650 -80 280 pi-0.5 0.3 -0.5];

%%
for (i = 1:size(CalibrationPositions,1))
   urMoveL(robot,CalibrationPositions(i,:));
    pause(2);
    img = snapshot(cam);    
    imwrite(img, ['calib/' num2str(i) '.bmp'], 'bmp')
    imshow(img);
end

%%
moverobot(robot, 1, [T O]);
save('calib/CalibrationPositions.mat', 'CalibrationPositions');

%% Execute calibration
cd('calib');
data_calib;
click_calib;
go_calib_optim;
saving_calib;
ext_calib;
cd('..');