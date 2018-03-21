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
subj_eye = 36;
img = database{subj_eye,2}{img_numb};

%img = imread('E:\2. P8 Project\Project\matlab_code\warsaw\0001right\session1\IMG_0102.jpg');
img = img(:,:,1);  % using the red band of the image as they did in the paper. CANNOT BE A DOUBLE

figure('Name','Whole eye'), colormap(gray(256));
imagesc(img);
%rmin, rmax:the minimum and maximum values of the iris radius. They have so
%far been chosen arbitrarily.
rmin = 80;
rmax = 180;
[ci,cp,out] = thresh(img,rmin,rmax);
figure('Name','Marked Eye'), imshow(out,[])
%OUTPUTS:
%cp:the parametrs[xc,yc,r] of the pupilary boundary
%ci:the parametrs[xc,yc,r] of the limbic boundary
%out:the segmented image

%% Eye Lid Suppresion

%imagewithnoise = neweyelidsup(img);
imagewithnoise = double(img);

%% Rubber Model Normalization
w = cd;
radial_res = 64;
angular_res = 512;

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
figure('Name','Polar Array')
colormap(gray(256));
imshow(polar_array);
%% Eyelash Removal

HistoFrac = 0.1;
RecognitionValue=2;

[counts,binLocations] = imhist(polar_array);

figure, stem(binLocations,counts);
Numberofbins=size(binLocations);

lowVal = 1.0;
HigVal = 0.0;

for i=1:1:Numberofbins(1)%Find the higest and the lovest binvalue of the histogram
    if counts(i)>RecognitionValue
        if binLocations(i)<lowVal
            lowVal=binLocations(i);
        end
        if binLocations(i)>HigVal
            HigVal=binLocations(i);
        end
    end
end

ThresVal=lowVal+HistoFrac*(HigVal-lowVal);%Find the threshold value based on the interval of the main histogram

reconstructIris=polar_array;
[polarrows,polarcols]=size(polar_array);
Equalised=zeros(polarrows,polarcols);
ref = polar_array < ThresVal;
[rows,cols] = find(ref==1);
processMap=ref;
NumberofEliminations=size(rows);
numberofUneliminatedNeighbors=0;
pixelVal=0;
SumVal=0;

UnprocessedPixels=NumberofEliminations(1);

while UnprocessedPixels>0
    
    for ii=1:1:NumberofEliminations(1)
        if processMap(rows(ii),cols(ii))==1
            if rows(ii)-1~=0
                if processMap(rows(ii)-1,cols(ii)) ~= 1 && isnan(polar_array(rows(ii)-1,cols(ii))) == 0
                    SumVal=SumVal+polar_array(rows(ii)-1,cols(ii));
                    numberofUneliminatedNeighbors = numberofUneliminatedNeighbors+1;
                    %alternative: add the values directly to the sum and devide by
                    %counter in the end
                end
            end
            if rows(ii)+1<=polarrows
                if processMap(rows(ii)+1,cols(ii)) ~= 1 && isnan(polar_array(rows(ii)+1,cols(ii))) == 0
                    SumVal=SumVal+polar_array(rows(ii)+1,cols(ii));
                    numberofUneliminatedNeighbors = numberofUneliminatedNeighbors+1;
                end
            end
            if cols(ii)-1~=0
                if processMap(rows(ii),cols(ii)-1) ~= 1 && isnan(polar_array(rows(ii),cols(ii)-1)) == 0
                    SumVal=SumVal+polar_array(rows(ii),cols(ii)-1);
                    numberofUneliminatedNeighbors = numberofUneliminatedNeighbors+1;
                end
            end
            if cols(ii)+1<=polarcols
                if processMap(rows(ii),cols(ii)+1) ~= 1 && isnan(polar_array(rows(ii),cols(ii)+1)) == 0
                    SumVal=SumVal+polar_array(rows(ii),cols(ii)+1);
                    numberofUneliminatedNeighbors = numberofUneliminatedNeighbors+1;
                end
            end
            %the numbers in the if statement below represents the number of
            %included
            if numberofUneliminatedNeighbors==4 || numberofUneliminatedNeighbors==3 || numberofUneliminatedNeighbors==2
                pixelVal=SumVal/numberofUneliminatedNeighbors;
                reconstructIris(rows(ii),cols(ii))=pixelVal;
                processMap(rows(ii),cols(ii))=0;
                UnprocessedPixels=UnprocessedPixels-1;
            end
            SumVal=0;
            numberofUneliminatedNeighbors=0;
            
        end
    end
end

figure('Name','ref'), imshow(ref)
figure('Name','Reconstructed iris'), imshow(reconstructIris)

%% Histogram Equalization

[countsN,binLocationsN] = imhist(reconstructIris); 

%figure, stem(binLocationsN,countsN);
NumberofbinsN=size(binLocationsN);

lowValN = 1.0;
HigValN = 0.0;

for iii=1:1:NumberofbinsN(1)%Find the higest and the lovest binvalue of the histogram
    if countsN(iii)>0
        if binLocationsN(iii)<lowValN
            lowValN=binLocationsN(iii);
        end
        if binLocationsN(iii)>HigValN
            HigValN=binLocationsN(iii);
        end
    end
end


for k=1:1:NumberofbinsN(1)
    temp=(binLocationsN(k)-lowValN)*(1/(HigValN-lowValN));
    if temp>0 && temp<1
      binLocationsNn(k)=temp;
      countsNn(k)=countsN(k);
    end
end

Equalised=(reconstructIris-lowValN)*(1/(HigValN-lowValN));

figure('Name','Equalised'), imshow(Equalised)

%% Feature Extraction


disp("finished running script");







