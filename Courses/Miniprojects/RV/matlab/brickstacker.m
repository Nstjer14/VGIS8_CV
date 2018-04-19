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


%% Connect to UR 
addpath(genpath('ur_interface'));
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

%% Get to home position. 
homepos = [156.8230 -387.7130  560.1100   -3.1296    0.0074    0.0004];
rotation = [-3.1296    0.0074    0.0004];
currentPos = urReadPosC(sock);
if isequal(currentPos,not(homepos)) == 0
    urMoveL(sock,homepos);
   
end
pause(2);
figure(1);
camshot = snapshot(cam);
imshow(camshot);

%% Calculate projection matrix
IP1 = [61 1024];
IP2 = [59 232];
IP3 = [863 230];

RP1 = [19.03 -678.12];
RP2 = [-160.92 -496];
RP3 = [22.58 -316.86];

X_rob = [RP1(1) RP2(1) RP3(1)]';
Y_rob = [RP1(2) RP2(2) RP3(2)]';

img_mat = [1 IP1; 1 IP2; 1 IP3;];
theta = linsolve(img_mat,X_rob); % X coordinate coefficients
phi = linsolve(img_mat,Y_rob); % Y coordinate coefficients
%% Get background images

disp('Remove bricks from table if any. Press enter when ready and clear.');
pause();

% Take background image
background = snapshot(cam);
figure(1)
imshow(background);
imwrite(background, 'background.bmp');

disp('Prepare bricks on table. Press enter when ready and clear.');
pause();
rgbImage = snapshot(cam);
figure(2);
imshow(rgbImage);
imwrite(rgbImage, 'rgbImage.bmp');
%imwrite(homeimg, ['calibrationfolder/' '3pt_calibration' '.bmp'], 'bmp')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------- config  --------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
configStruct =  struct('I_treshold_min',uint8(20),...
                       'red_min',[],... %[x_min x_max y_min y_max]
                       'red_max',[],...
                       'green_min',[],...
                       'green_max',[],...
                       'R_closing',10,...
                       'R_opening',5,...
                       'LowerPrcTile',5,...
                       'upperPrcTile',95,...
                       'color',[],...
                       'mass_min',1000);
                   
% Brick dimensions
BrickHeight = 20;
BrickWidth = 27;

%% Setup Vision processing configuration                   
RB_config = configStruct;
GB_config = configStruct;
BB_config = configStruct;
YB_config = configStruct;
OB_config = configStruct;
%WB_config = configStruct;

RB_config = configStruct;
RB_config.name = 'Red';
GB_config = configStruct;
GB_config.name = 'Green';
BB_config = configStruct;
BB_config.name = 'Blue';
YB_config = configStruct;
YB_config.name = 'Yellow';
OB_config = configStruct;
OB_config.name = 'Orange';
%WB_config = configStruct;
%WB_config.name = 'White';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------- analysis of threshold values  ---------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('thresholds.mat');

RB_config.red_min = RB_red_min;
RB_config.red_max = RB_red_max;
RB_config.green_min = RB_green_min;
RB_config.green_max = RB_green_max;
clear RB_red_min RB_red_max RB_green_min RB_green_max

GB_config.red_min = GB_red_min;
GB_config.red_max = GB_red_max;
GB_config.green_min = GB_green_min;
GB_config.green_max = GB_green_max;
clear GB_red_min GB_red_max GB_green_min GB_green_max

BB_config.red_min = BB_red_min;
BB_config.red_max = BB_red_max;
BB_config.green_min = BB_green_min;
BB_config.green_max = BB_green_max;
clear BB_red_min BB_red_max BB_green_min BB_green_max

YB_config.red_min = YB_red_min;
YB_config.red_max = YB_red_max;
YB_config.green_min = YB_green_min;
YB_config.green_max = YB_green_max;
clear YB_red_min YB_red_max YB_green_min YB_green_max

%WB_config.red_min = WB_red_min;
%WB_config.red_max = WB_red_max;
%WB_config.green_min = WB_green_min;
%WB_config.green_max = WB_green_max;
%clear WB_red_min WB_red_max WB_green_min WB_green_max

OB_config.red_min = OB_red_min;
OB_config.red_max = OB_red_max;
OB_config.green_min = OB_green_min;
OB_config.green_max = OB_green_max;
clear OB_red_min OB_red_max OB_green_min OB_green_max

mass_min = mass_min/3;
RB_config.mass_min = mass_min;
GB_config.mass_min = mass_min;
BB_config.mass_min = mass_min;
YB_config.mass_min = mass_min;
%WB_config.mass_min = mass_min;
OB_config.mass_min = mass_min;
clear mass_min;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------- image processing ----------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rgiImage = RGB2RGI2( rgbImage );
imwrite(rgbImage, [TimeStamp '.bmp']);

%% Background Subtraction

BG_threshold = 12;

ForegroundImage = BackgroundSubtraction( rgiImage,BG_threshold );

rgiImage = ForegroundImage;

RB = imageProcessing( rgiImage,RB_config ); % Returns cell array with 
                                            % information of each brick 
                                            % in the given color
GB = imageProcessing( rgiImage,GB_config );
BB = imageProcessing( rgiImage,BB_config );
YB = imageProcessing( rgiImage,YB_config );
%WB = imageProcessing( rgiImage,WB_config );
OB = imageProcessing( rgiImage,OB_config );

saveImages2

%PlaceX = 400;
%PlaceY = 120;
%PlaceZ = BrickPlaceZ;
%RobotPlaceLocation = [PlaceX, PlaceY, PlaceZ, pi, 0, 0];
%BricksPickedUp = 1;
Bricks = {RB, GB, BB, YB, OB}; % WB removed

%% Move robot to brick
disp('Get ready to the robot to get funky!. Press enter when ready and clear.');
pause();
brickPos = Bricks{1,1}{1}.boudingBoxCenter;
zHeight = 144.8210;
worldBrickPos = [[1 brickPos] * [theta phi] zHeight+BrickHeight];


ToolPose = GetSO4FromURpose(homepos);
BrickPose = ToolPose * trotz(deg2rad(-Bricks{1,1}{1}.rotation));
BrickPose(1:3,4) = worldBrickPos;
BrickPoseUR = GetURposeFromSO4(BrickPose);

urMoveL(sock,BrickPoseUR);

%deg2rad(Bricks{1,1}{1}.rotation);
%rotation = [-3.1296    0.0074    0.0004];




