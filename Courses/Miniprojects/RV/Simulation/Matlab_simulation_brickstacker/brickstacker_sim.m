clear all;

% myConnector.movePTP(robothome,'v100');
clc;
close all;
addpath(genpath('D:\2. P8 Project\Courses\Miniprojects\RV\Simulation\Tutorial 3 - Matlab'));

myConnector = RobotStudioConnector('127.0.0.1',1024);
robothome = [236.7230  282.1160  601.8670    0.0076    0.0002    0.9993   -0.0366]; % in quartionions
homepos = [440.8810   56.2063  254.1650  0 0 1 0]; %[robothome(1:3) quat2eul(robothome(4:end))];
pickupOrientation = [440.8810   56.2063  254.1650  0 0 1 0];
pickupOrientation_eul = [pickupOrientation(1:3) deg2rad(quat2eul(pickupOrientation(4:end)))];
dropOffPos = [411.0700 -300.0000  244.8200    0.0000   -0.0000    1.0000   -0.0000];

%myConnector.gripperOn()
%%
%myConnector.movePTP(448.2070,300.0000,15.3523,0.3256,0.0000,-0.9455,-0.0000,'v100');
%%
B = myConnector.getPosition();
B
%myConnector.gripperOff();
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

% There are the X,Y coordinates of the bricks in world space. They are the
% calibration points
% calibrationImage = imread('withBlocks.png');
%P1 = [513.144 132.435] 
%P2 = [525.361 -42.115]
%P3 = [361.276 75.068]

% zHeight = 254.1650;
% Orientation = [0.0002    0.9993   -0.0366];
IP1 = [221 226];
IP2 = [775 183];
IP3 = [404 721];

RP1 = [513.1400  132.4300];
RP2 = [525.3600  -42.1100];
RP3 = [361.2800   75.0680];

X_rob = [RP1(1) RP2(1) RP3(1)]';
Y_rob = [RP1(2) RP2(2) RP3(2)]';

img_mat = [1 IP1; 1 IP2; 1 IP3;];
theta = linsolve(img_mat,X_rob); % X coordinate coefficients
phi = linsolve(img_mat,Y_rob); % Y coordinate coefficients
% example of how to use: fromPixelToWorld = [1 IP] * [theta phi];
% newPointInWorld = [fromPixelToWorld Z Orientation]
%fromPixelToWorld = [1 IP1] * [theta phi];
%newPointInWorld = [fromPixelToWorld zHeight+20 0.0076    0.0002    0.9993   -0.0366];
%myConnector.movePTP(newPointInWorld,'v100');
%myConnector.gripperOn();
%pause(3);
%% Get images for background subtraction
%figure(1)
background_png = imread('background.png');
imwrite(background_png,'background.bmp','bmp');
background = imread('background.bmp');

%imshow(background);

%figure(2);
withBlocks = imread('withBlocks.png');
imwrite(withBlocks,'rgbImage.bmp','bmp');
rgbImage =  imread('rgbImage.bmp');
%imshow(rgbImage);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------------------------- config  --------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
configStruct =  struct('I_treshold_min',uint8(12),...
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
BrickHeight = 11;
BrickWidth = 16;

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

figure(3);
imshow(ForegroundImage);

%% Move robot to brick
zHeight = 234.8200;
disp('Enter the character you want made!: ');
pause();

RL = length(Bricks{1,1});
GL = length(Bricks{1,2});
BL = length(Bricks{1,3});
YL = length(Bricks{1,4});
OL = length(Bricks{1,5});

dropoff = [-259.16 -235.14 220.17 2.1095 -2.2545 0.0202];

while(1)
    pause();
    char = input('Enter the character you want made!: ' , 's');
    switch char
        case 'Homer'
            run('homer.m');
            %urMoveL(sock,dropoff);
            %myConnector.movePTP(homepos,'v100');;
        case 'Marge'
            run('marge.m');
            %urMoveL(sock,dropoff);
            %myConnector.movePTP(homepos,'v100');;
        case 'Bart'
            run('bart.m');
            %urMoveL(sock,dropoff);
            %myConnector.movePTP(homepos,'v100');;
        case 'Lisa'
            run('lisa.m');
            %urMoveL(sock,dropoff);
            %myConnector.movePTP(homepos,'v100');;
        case 'Maggie'
            run('maggie.m');
            %urMoveL(sock,dropoff);
            %myConnector.movePTP(homepos,'v100');;
        case 'Exit'
            break;
            
        otherwise
            disp('Character unknonw');
            
    end
end
