%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Group ID : 842
% Members : Shagen, Marike, Niclas
% Date : 2018-03-21
% Dependencies: libormasek, Daugmans Integrodifferential Operator,
% folder named diagnostics, warsaw database, Voicebox, Wavelet Toolbox
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

showFigure = false;
extractedDatabase = [];
classifier = [];
%% Performing Daugmans Integro Differential operator to locate the center of


for ii = 1:length(database)
    subj_numb = ii;
    skip = false;
    if isempty(database{subj_numb,2}) == true
        %subj = database{subj_numb,1};
        %disp(subj);
        skip = true;
    end
    if skip == false;
        for jj = 1:length(database{subj_numb,2})
            img_numb = jj;
        %    disp(img_numb);


%subj_numb = 48;
% img_numb = 1;
img = database{subj_numb,2}{img_numb};

%img = imread('E:\2. P8 Project\Project\matlab_code\warsaw\0001right\session1\IMG_0102.jpg');
img = img(:,:,1);  % using the red band of the image as they did in the paper. CANNOT BE A DOUBLE

if showFigure == true
    figure('Name','Whole eye'), colormap(gray(256));
    imagesc(img);
end
%rmin, rmax:the minimum and maximum values of the iris radius. They have so
%far been chosen arbitrarily.
rmin = 80;
rmax = 180;
[ci,cp,out] = thresh(img,rmin,rmax);
if showFigure == true
    figure('Name','Marked Eye');
    imshow(out,[]);
end
%OUTPUTS:
%cp:the parametrs[xc,yc,r] of the pupilary boundary
%ci:the parametrs[xc,yc,r] of the limbic boundary
%out:the segmented image

%% Eye Lid Suppresion

imagewithnoise = neweyelidsup(img);
%imagewithnoise = double(img);
if showFigure == true
    figure('Name','Supressed Eyelids');
    imshow(imagewithnoise,[]);
end
%% Rubber Model Normalization
w = cd;
radial_res = 64;
angular_res = 512;

%mkdir(database{2,1});
eyeimage_filename = strcat(database{subj_numb,1},'_',num2str(img_numb));

[polar_array noise_array] = normaliseiris((imagewithnoise),...
    ci(2), ci(1), ci(3),...
    cp(2), cp(1), cp(3),...
    eyeimage_filename, radial_res, angular_res);


cd(DIAGPATH);
imwrite(polar_array,[eyeimage_filename,'-polar.jpg'],'jpg');
imwrite(noise_array,[eyeimage_filename,'-polarnoise.jpg'],'jpg');
cd(w); % Return to the script folder

if showFigure == true
    %Show the normalised image
    figure('Name','Polar Array')
    colormap(gray(256));
    imshow(polar_array);
end
%% Eyelash Removal

HistoFrac = 0.1;
RecognitionValue=2;

[reconstructIris] = noiseremover(polar_array,HistoFrac,RecognitionValue);
if showFigure == true
    figure('Name','Reconstructed iris');
    imshow(reconstructIris,[]);
end

%% Histogram Equalization

[equalised]=equalisehistogram(reconstructIris);

if showFigure == true
    figure('Name','Equalised iris')
    imshow(equalised,[]);
end

%% Feature Extraction

[a,h,v,d] = haart2(equalised,3);
if showFigure == true
    figure('Name','Haar Wavelets')
    %colormap(gray(256));
    imshow(a);
end
%a is the same as LL is called the approximation of the image.
%h is the same as LH is the horizontal detail,
%v is the same as HL is the vertical detail
%d is the same as HH represents the diagonal detail of the image.

feature_vec = reshape(a',[1,512]); % Resizing as they did in the paper. The rows are concatanated.

disp("finished running script");
extractedDatabase = vertcat(extractedDatabase,feature_vec);
classifier = vertcat(classifier,string(database{subj_numb,1}));
disp(subj_numb);
disp(img_numb);
        end
    end
end

save('database_segmented.mat','extractedDatabase','classifier','-v7.3');




