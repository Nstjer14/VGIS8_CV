%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Group ID : 842
% Members : Shagen, Marike, Niclas
% Date : 2018-03-21
% Dependencies: libormasek, Daugmans Integrodifferential Operator,
% folder named diagnostics, warsaw database Voicebox
% Matlab version: R2017b
% Functionality: Using Daugmans Integrodifferential Operator to find the
% iris and pupil bounds, masek to supress the eyelids and use daugmans
% rubber sheet model to normalise the image. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clearvars -except database
clc;
close all;

% Check of database exist. If not load it.
if ~exist('database')
    database = load('database.mat');
    database = database.database;
end 

addpath(genpath('libormasek'));
addpath(genpath('daugman'));

% Adding global variable to comply with libormasek scripts
global DIAGPATH
DIAGPATH = 'diagnostics';

%% Performing Daugmans Integro Differential operator to locate the center of

img_numb = 2;
subj_eye = 1;
img = database{subj_eye,2}{img_numb};
img = double(img(:,:,1)); % using the red band of the image as they did in the paper.

%rmin, rmax:the minimum and maximum values of the iris radius. They have so
%far been chosen arbitrarily.
rmin = 30;
rmax = 220;
[ci,cp,out] = thresh(img,rmin,rmax);

%OUTPUTS:
%cp:the parametrs[xc,yc,r] of the pupilary boundary
%ci:the parametrs[xc,yc,r] of the limbic boundary
%out:the segmented image

%% Eye Lid Suppresion

imagewithnoise = neweyelidsup(img);

%% Rubber Model Normalization
w = cd; 
radial_res = 20;
angular_res = 240;

%mkdir(database{2,1});
eyeimage_filename = strcat(database{subj_eye,1},'_',num2str(img_numb));

[polar_array noise_array] = normaliseiris((imagewithnoise),...
    ci(2), ci(1), ci(3),...
    cp(2), cp(1), cp(3),...
    eyeimage_filename, radial_res, angular_res);


cd(DIAGPATH);
imwrite(polar_array,[eyeimage_filename,'-polar.jpg'],'jpg');
imwrite(noise_array,[eyeimage_filename,'-polarnoise.jpg'],'jpg');
cd(w); % Return to the script folder

%Show the normalised image
colormap(gray(256));
imshow(polar_array);
%% Eyelash Removal

%% Histogram Equalization

%% Feature Extraction


disp("finished running script");







