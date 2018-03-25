%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Date: 2018-03-18
% Dependencies:    Computer Vision Toolbox, Daugman integro scripts,
%                  libormasek scripts.
% Version:         Matlab R2017b
% Functionnallity: Parses through the Iris Database from Warsaw. The
%                  database is stored in a cell array where database{:,1} is the name and 
%                  database{:,2} are the images. An image can be accesed by doing
%                  database{:,2}{n}. There are 70 subjects with 2 iris (left and right)
%                  each. Subject 24, 29 and 62 (left) are empty folders with no images.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear all;
clc;

addpath(genpath('daugman'));
addpath(genpath('libormasek'));

database_name = 'warsaw';
files = dir(database_name); % Get information about the folder structure
directoryNames = {files([files.isdir]).name}; % Get all the individual names.
n = length(directoryNames)-2; % The first two entries are for some reason empty, so they have to be ignored.

database = cell(n,2);
tic
for ii=1:n;
    database{ii,1} = directoryNames{ii+2}; % Storing subject name
    imgSetVector = imageSet(strcat(database_name,'\',directoryNames{ii+2}),'recursive'); % From Computer Vision Toolbox.
    
    for jj = 1:length(imgSetVector); % Parsing through the two subfolders for session 1 and 2
        subimage_names = imgSetVector(jj).ImageLocation;
       
        for kk = 1:length(imgSetVector(jj).ImageLocation); % Parsing through folder with image and reading them
            tempimg{kk} = imread(subimage_names{kk});
        end
        database{ii,2}{end+1} = tempimg;
        clearvars tempimg
    end
    
    % If there are two folder it creates two cells. This concatantes them
    % into a single cell aray.
    if (length(database{ii,2}) == 2) 
        database{ii,2} = [database{ii,2}{1},database{ii,2}{2}];
    else if (length(database{ii,2}) == 1)
            database{ii,2} = database{ii,2}{1};
        end
    end    
end
toc
save('database.mat','database','-v7.3');
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OBS! BELOW IS OLD SHIT CODE THAT IS NOT RELEVANT BUT CAN BE USEFULL TO
% SHAGEN LATER. JUST IGNORE EVERYTHING BELOW.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% pth = genpath(database_name);
% pth = strsplit(pth, ';') ;
% listing =  dir(database_name);
% 
% data(140)=struct('name',[],'images',[])
% 
% name = listing(6).name;
% filetype = '*.JPG';
% folder_name = pth{3};
% imagefiles = dir(strcat(folder_name,'\',filetype));      
% nfiles = length(imagefiles);    % Number of files found
% 
% for ii=1:nfiles
%    currentfilename = imagefiles(ii).name;
%    currentimage = imread(strcat(folder_name,'\',currentfilename));
%    images{ii} = currentimage;
% end
% 
% data(1,1).name = name;
% data(1,1).images = images;
% fp1 = {}