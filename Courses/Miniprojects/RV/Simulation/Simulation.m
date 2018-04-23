%clear all;
%clc;
addpath(genpath('D:\2. P8 Project\Courses\Miniprojects\RV\Simulation\Tutorial 3 - Matlab'));

myConnector = RobotStudioConnector('127.0.0.1',1024);
%myConnector.gripperOn()
%%
%myConnector.movePTP(448.2070,300.0000,15.3523,0.3256,0.0000,-0.9455,-0.0000,'v100');
%%
%B = myConnector.getPosition();
myConnector.gripperOff();
%%
%myConnector.takePicture();
%%
%myConnector.moveJoints(0,0,0,0,0,0,'v100');

%%
%myConnector.movePTP(0,0,0,0,0,0,0,'v100');
%%
%myConnector.moveRelativePTP(0,100,0,0,0,0,0,'v100');
%%
%myConnector.moveRelativeLinear(0,0,0,0,0,0,0,'v100');
%
%% Calculate matrix to get from image to robot world

IP1 = [272 331];
IP2 = [271 221];
IP3 = [438 220];

RP1 = [198.6750  300.0000];
RP2 = [448.2070  300.0000];
RP3 = [448.2070  -77.6290];

dXIP = IP3(1)-IP2(1);
dYIP = IP2(2)-IP1(2);

dXRP = RP3(1)-RP2(1);
dYRP = RP2(2)-RP1(2);

dX = dXRP/dXIP;
dY = dYRP/dYIP;

O = [0.3256    0.0000   -0.9455   -0.0000];
T = [IP1(1)*dX IP1(2)*dY 15.3523];
WP = [T O];
myConnector.movePTP(WP,'v100');

%% To do
% Add lego blocks
% Add Camera
% Add gripper
% Add recognition system