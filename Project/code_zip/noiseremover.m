%%%
%The input value HistoFrac is how big a faction of the identified present
%histogram spectrum is filtered out. The input value RecognitionValue defines 
%what the pixelcount in a given histogram bin has to be larger than before it is
%considered as a present value.
%%%

function [reconstructIris] = noiseremover(sourceimage,HistoFrac,RecognitionValue)


%HistoFrac = 0.1;
%RecognitionValue=2;
[counts,binLocations] = imhist(sourceimage);

%figure, stem(binLocations,counts);
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

reconstructIris=sourceimage;
[polarrows,polarcols]=size(sourceimage);
Equalised=zeros(polarrows,polarcols);
ref = sourceimage < ThresVal;
[rows,cols] = find(ref==1);
processMap=ref;
NumberofEliminations=size(rows);
numberofUneliminatedNeighbors=0;
pixelVal=0;
SumVal=0;

UnprocessedPixels=NumberofEliminations(1);

 while UnprocessedPixels>0 %while there are still unreconstructed pixels

for ii=1:1:NumberofEliminations(1) 
    if processMap(rows(ii),cols(ii))==1
    if rows(ii)-1~=0    
    if processMap(rows(ii)-1,cols(ii)) ~= 1 && isnan(sourceimage(rows(ii)-1,cols(ii))) == 0 
       SumVal=SumVal+reconstructIris(rows(ii)-1,cols(ii));
        numberofUneliminatedNeighbors = numberofUneliminatedNeighbors+1;
    end
    end
    if rows(ii)+1<=polarrows
    if processMap(rows(ii)+1,cols(ii)) ~= 1 && isnan(sourceimage(rows(ii)+1,cols(ii))) == 0 
        SumVal=SumVal+reconstructIris(rows(ii)+1,cols(ii));
      numberofUneliminatedNeighbors = numberofUneliminatedNeighbors+1;
    end
    end
    if cols(ii)-1~=0
    if processMap(rows(ii),cols(ii)-1) ~= 1 && isnan(sourceimage(rows(ii),cols(ii)-1)) == 0 
        SumVal=SumVal+reconstructIris(rows(ii),cols(ii)-1);
      numberofUneliminatedNeighbors = numberofUneliminatedNeighbors+1;
    end
    end
    if cols(ii)+1<=polarcols
    if processMap(rows(ii),cols(ii)+1) ~= 1 && isnan(sourceimage(rows(ii),cols(ii)+1)) == 0 
        SumVal=SumVal+reconstructIris(rows(ii),cols(ii)+1);
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