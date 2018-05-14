

function [Equalised]=equalisehistogram(reconstructIris) 


[polarrows,polarcols]=size(reconstructIris);
Equalised=zeros(polarrows,polarcols);

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