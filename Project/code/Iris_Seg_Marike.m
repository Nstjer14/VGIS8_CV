
close all
clear all

HistoFrac = 0.1;
RecognitionValue=2;

path='/Users/Marike/Documents/MATLAB/Iris_database/Warsaw-BioBase-Smartphone-Iris-v1.0/0018right/session1/IMG_0423.jpg';

global DIAGPATH
DIAGPATH = 'iriscode/diagnostics';
filename=dir(path);
disp(filename.name);

I=imread(path);

%imshow(I)

%H=rgb2gray(I);
G=I(:,:,1);
%H=I(:,:,2);
%Z=I(:,:,3);
%figure, imshow(H);
%figure, imshow(G);
%figure, imshow(Z);
%imwrite(G,'Test.jpg');
[ci,cp,marked]=thresh(G,80,180);

figure, imshow(marked)

radial_res = 20;
angular_res = 240;

[polar_array, polar_noise]=normaliseiris(double(G),ci(2),ci(1),ci(3),cp(2),cp(1),cp(3),filename.name, radial_res, angular_res);

figure, imshow(polar_array)

[counts,binLocations] = imhist(polar_array); 
%hist(polar_array,256);

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

figure, imshow(ref)
figure, imshow(reconstructIris)

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

figure, imshow(Equalised)
%figure, stem(binLocationsNn',countsNn);
