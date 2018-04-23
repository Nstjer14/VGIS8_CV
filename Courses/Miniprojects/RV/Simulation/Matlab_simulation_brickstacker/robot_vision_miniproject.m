%% Setup
clear;
clc;
addpath(genpath('ur_interface'));
addpath(genpath('thomas_lib'));


% My static IP is set to 192.168.0.20
% Get the robot ip from the teach pendant: File -> About
robot_ip = '192.168.0.2';
%  robot_ip = '127.0.0.1';         % URsim

sock = tcpip(robot_ip, 30000, 'NetworkRole', 'server');
fclose(sock);
disp('Press Play on robot');
fopen(sock);
disp('Connected!');

%% 
fprintf('Reading initial pose...\n');
posC = urReadPosC(sock);
fprintf('posC = ');disp(posC);
posJ = urReadPosJ(sock);
fprintf('posJ = ');disp(posJ);


% Change velocity and acceleration:
urChangeVel(sock, [0.1,0.1]);

% Move to new joint position:
%posJ(3) = posJ(3) + 0.15;
%urMoveJ(sock,posJ);

% Open pneumatic gripper:
urSetIO(sock, 0, 1);
urSetIO(sock, 1, 0);