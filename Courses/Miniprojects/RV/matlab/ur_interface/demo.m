%% Description
% This script is to be used together with a UR script, running on the
% UR-controller, called "matlab_rsa_x.x.urp", where x.x is the version no.
% Currently, this script has been tested in ursim with:
%  - matlab_rsa_3.4.urp works with:
%       - ursim 3.4 and 3.5
%       - UR5 3.3.4.310
%       - UR5 3.5.0.10584
%  - matlab_rsa_3.3.urp: Not testet
%  - matlab_rsa_3.0.urp: Does not work (ursim-3.0 cannot open any of the files)
%  - last_test_1.8.urp : ursim-1.8 does not work
%
% GUIDE
% 1) Copy "matlab_rsa_x.x.urp" to the UR robot (using USB)
% 2) Open demo.m on the PC and "matlab_rsa_x.x.urp" on the UR controller
% 3) IP adddresses
%    a) Setup static IP on the PC and try to ping the UR controller
%    b) Correct 'robot_ip' in the matlab script 
%    c) Correct the IP on the UR controller in the line 'socket_open(...)'
% 4) Start demo.m
% 5) When it says 'Press Play on robot', do it
%    The robot will now move, so BE READY WITH THE EMERGENCY STOP!

%% Setup
clear;
clc;



% Get the robot ip from the teach pendant: File -> About
robot_ip = '192.168.0.2';
%  robot_ip = '127.0.0.1';         % URsim

sock = tcpip(robot_ip, 30000, 'NetworkRole', 'server');
fclose(sock);
disp('Press Play on robot');
fopen(sock);
disp('Connected!');

%% Move

% Read pose:
fprintf('Reading initial pose...\n');
posC = urReadPosC(sock);
fprintf('posC = ');disp(posC);
posJ = urReadPosJ(sock);
fprintf('posJ = ');disp(posJ);

%%

for i = 1:3
    fprintf('Moving (%d)...\n',i);
    
    % Change velocity and acceleration:
    urChangeVel(sock, [0.1,0.1]);
    
    % Move to new joint position:
    posJ(3) = posJ(3) + 0.15;
    urMoveJ(sock,posJ);
    
    % Open pneumatic gripper:
    urSetIO(sock, 0, 1);
    urSetIO(sock, 1, 0);
    
    % Change velocity and acceleration:
    urChangeVel(sock, [0.4,0.4]);
    
    % Move to new joint position:
    posJ(3) = posJ(3) - 0.15;
    urMoveJ(sock,posJ)
    
    % Close pneumatic gripper:
    urSetIO(sock, 0, 0);
    urSetIO(sock, 1, 1);
end

% Close connection:
% fclose(sock);
