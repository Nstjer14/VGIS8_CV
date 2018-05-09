import numpy as np
import matplotlib.pyplot as plt
import copy

#@profile
def equalisehistogram(reconstructIris,LimitValue):

    numberOfBins=256
    histE, bin_edgesE=np.histogram(reconstructIris, numberOfBins, range=(0,256), density=False) #create histogram
    imageDim=reconstructIris.shape 
    lowVal = 255.0
    higVal = 0.0

    for i in range(0,numberOfBins): #find the lowest and highest bin values where the frequency is higher than a specified LimitValue 
        if histE[i]>LimitValue:
            if bin_edgesE[i]<lowVal:
                lowVal=bin_edgesE[i]
            if bin_edgesE[i]>higVal:
                higVal=bin_edgesE[i]

    Equalised=(reconstructIris-lowVal)*(255/(higVal-lowVal))
    withoutovers=np.where(Equalised>255,255,Equalised)
    withoutunders=np.where(withoutovers<0,0,withoutovers) 

    return withoutunders



#@profile
def noiseremover(sourceimage,HistoFrac,RecognitionValue): 

    numberOfBins=256
    hist, bin_edges=np.histogram(sourceimage, numberOfBins, range=(0,255), density=False)
    lowVal = 255.0
    higVal = 0.0
    for i in range(0,numberOfBins):#find the lowest and highest bin values where the frequency is higher than a specified RecognitionValue
        if hist[i]>RecognitionValue:
            if bin_edges[i]<lowVal:
                lowVal=bin_edges[i]
            if bin_edges[i]>higVal:
                higVal=bin_edges[i]
    ThresVal=lowVal+HistoFrac*(higVal-lowVal);
    reconstructIris=copy.deepcopy(sourceimage)
    imageDim=sourceimage.shape
    ref=np.empty(imageDim, dtype=bool)
    ref = sourceimage < ThresVal
    Coordinates = np.where(ref==True)
    processMap=copy.deepcopy(ref)
    NumberofEliminations=len(Coordinates[0])
    numberofUneliminatedNeighbors=0
    pixelVal=0
    SumVal=0
    UnprocessedPixels=copy.deepcopy(NumberofEliminations)

    while UnprocessedPixels>0: #While there are still pixels that have not been reconstructed 
        for ii in range(0,NumberofEliminations): #go through all of the eliminated pixels
            if processMap[Coordinates[0][ii]][Coordinates[1][ii]]==True: #if the current pixel still has not been reconstructed then do reconstruction
                if Coordinates[0][ii]-1>=0: #look at whether the neighbor pixel exist within the image boundary 
                    if processMap[Coordinates[0][ii]-1][Coordinates[1][ii]] == False and sourceimage[Coordinates[0][ii]-1][Coordinates[1][ii]] is not None: #make sure the neighbor pixel is not none and does not need reconstruction.
                        SumVal = SumVal + reconstructIris[Coordinates[0][ii] - 1][Coordinates[1][ii]] #
                        numberofUneliminatedNeighbors = numberofUneliminatedNeighbors+1
                if Coordinates[0][ii]+1<imageDim[0]:#make sure the neighbor is within the image boundary
                    if processMap[Coordinates[0][ii]+1][Coordinates[1][ii]] == False and sourceimage[Coordinates[0][ii]+1][Coordinates[1][ii]] is not None:
                        SumVal = SumVal + reconstructIris[Coordinates[0][ii] + 1][Coordinates[1][ii]]
                        numberofUneliminatedNeighbors = numberofUneliminatedNeighbors+1
                if Coordinates[1][ii]-1>=0: #make sure the neighbor is within the image boundary
                    if processMap[Coordinates[0][ii]][Coordinates[1][ii] - 1] == False and sourceimage[Coordinates[0][ii]][Coordinates[1][ii] - 1] is not None:
                        SumVal = SumVal + reconstructIris[Coordinates[0][ii]][Coordinates[1][ii]-1]
                        numberofUneliminatedNeighbors = numberofUneliminatedNeighbors+1
                if Coordinates[1][ii] + 1 < imageDim[1]:#make sure the neighbor is within the image boundary
                    if processMap[Coordinates[0][ii]][Coordinates[1][ii] + 1] == False and sourceimage[Coordinates[0][ii]][Coordinates[1][ii] + 1] is not None: 
                        SumVal = SumVal + reconstructIris[Coordinates[0][ii]][Coordinates[1][ii] + 1]
                        numberofUneliminatedNeighbors = numberofUneliminatedNeighbors+1
                #the numbers in the if statement below represents the number of included 
                if numberofUneliminatedNeighbors == 4 or numberofUneliminatedNeighbors == 3 or numberofUneliminatedNeighbors == 2:  
                     pixelVal=SumVal/numberofUneliminatedNeighbors
                     reconstructIris[Coordinates[0][ii]][Coordinates[1][ii]]=pixelVal
                     processMap[Coordinates[0][ii]][Coordinates[1][ii]]=False
                     UnprocessedPixels=UnprocessedPixels-1
                SumVal=0
                numberofUneliminatedNeighbors=0
    return reconstructIris


##################For Testing###########################

if __name__ == '__main__':

    F=plt.imread( '/Users/Marike/Documents/MATLAB/iriscode/diagnostics/0002right_7-polar.jpg' )
    HistFrac=0.1
    RecVal=40
    
    
    #K=noiseremover(F,HistFrac,RecVal)
    #L=equalisehistogram(F,40)


