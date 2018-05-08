import numpy as np
import matplotlib.pyplot as plt
import copy


def customhist(image,numberofbins,ran):
    increment=ran[1]/numberofbins
    binsE=np.zeros(numberofbins+1)
    hist=np.zeros(numberofbins)
    for i in range(0,numberofbins+1):
         binsE[i]=i*increment
    imdim=image.shape
    for ii in range(0,imdim[0]):
        for iii in range(0,imdim[1]):
            for pp in range(0,numberofbins):
                if image[ii][iii]>=binsE[pp] and image[ii][iii]<binsE[pp+1]:
                    hist[pp]=hist[pp]+1
    return hist, binsE 

@profile
def equalisehistogram(reconstructIris,LimitValue):

    numberOfBins=256
    histE, bin_edgesE=customhist(reconstructIris,numberOfBins,(0,256))
    imageDim=reconstructIris.shape
    Equalised=np.zeros(imageDim)
    lowVal = 255.0
    higVal = 0.0

    for i in range(0,numberOfBins):
        if histE[i]>LimitValue:
            if bin_edgesE[i]<lowVal:
                lowVal=bin_edgesE[i]
            if bin_edgesE[i]>higVal:
                higVal=bin_edgesE[i]

    for p in range(0,imageDim[0]):
         for c in range(0,imageDim[1]):
              temp=(reconstructIris[p][c]-lowVal)*(255/(higVal-lowVal))
              Equalised[p][c]=temp
              if temp<0:
                  Equalised[p][c]=0 
              if temp>255:
                  Equalised[p][c]=255 
    return Equalised






#=np.histogram(reconstructIris,numberOfBins,range=(0,256),density=False)






@profile
def noiseremover(sourceimage,HistoFrac,RecognitionValue): 

    numberOfBins=256
    hist, bin_edges=customhist(sourceimage,numberOfBins,(0,256))
    #plt.hist(sourceimage.ravel(),256)
    #plt.show()

    lowVal = 255.0
    higVal = 0.0

    for i in range(0,numberOfBins):
        if hist[i]>RecognitionValue:
            if bin_edges[i]<lowVal:
                lowVal=bin_edges[i]
            if bin_edges[i]>higVal:
                higVal=bin_edges[i]

    ThresVal=lowVal+HistoFrac*(higVal-lowVal);
    reconstructIris=copy.deepcopy(sourceimage)
    imageDim=sourceimage.shape
    ref=np.empty(imageDim, dtype=bool)
    xCord=[]
    yCord=[]
    for h in range(0,imageDim[0]):
        for s in range(0,imageDim[1]):
            if sourceimage[h][s]<=ThresVal:
                ref[h][s]=True
                xCord.append(h)
                yCord.append(s)
            else:
                ref[h][s]=False
    processMap=copy.deepcopy(ref)
    NumberofEliminations=len(xCord)
    numberofUneliminatedNeighbors=0
    pixelVal=0
    SumVal=0
    UnprocessedPixels=NumberofEliminations

    
    while UnprocessedPixels>0:
        for ii in range(0,NumberofEliminations):
            if processMap[xCord[ii]][yCord[ii]]==True:#if the current pixel still has not been reconstructed then do reconstruction
                if xCord[ii]-1>=0:#make sure the neighbor is within the image boundary 
                    if processMap[xCord[ii]-1][yCord[ii]] == False and sourceimage[xCord[ii]-1][yCord[ii]] is not None: #make sure the neighbor pixel is not none and does not need reconstruction.
                        SumVal=SumVal+reconstructIris[xCord[ii]-1][yCord[ii]] #contribute to current pixel reconstruction 
                        numberofUneliminatedNeighbors = numberofUneliminatedNeighbors+1
                if xCord[ii]+1<imageDim[0]:#make sure the neighbor is within the image boundary
                    if processMap[xCord[ii]+1][yCord[ii]] == False and sourceimage[xCord[ii]+1][yCord[ii]] is not None: #make sure the neighbor pixel is not none and does not need reconstruction.
                        SumVal=SumVal+reconstructIris[xCord[ii]+1][yCord[ii]]
                        numberofUneliminatedNeighbors = numberofUneliminatedNeighbors+1
                if yCord[ii]-1>=0: #make sure the neighbor is within the image boundary
                    if processMap[xCord[ii]][yCord[ii]-1] == False and sourceimage[xCord[ii]][yCord[ii]-1] is not None: #make sure the neighbor pixel is not none and does not need reconstruction.
                        SumVal=SumVal+reconstructIris[xCord[ii]][yCord[ii]-1]
                        numberofUneliminatedNeighbors = numberofUneliminatedNeighbors+1
                if yCord[ii]+1<imageDim[1]:#make sure the neighbor is within the image boundary
                    if processMap[xCord[ii]][yCord[ii]+1] == False and sourceimage[xCord[ii]][yCord[ii]+1] is not None: #make sure the neighbor pixel is not none and does not need reconstruction. 
                        SumVal=SumVal+reconstructIris[xCord[ii]][yCord[ii]+1]
                        numberofUneliminatedNeighbors = numberofUneliminatedNeighbors+1
                #the numbers in the if statement below represents the number of included neighbors 
                if numberofUneliminatedNeighbors==4 or numberofUneliminatedNeighbors==3 or numberofUneliminatedNeighbors==2:  
                     pixelVal=SumVal/numberofUneliminatedNeighbors #calculate pixel value based on average of existing neighbor pixels
                     reconstructIris[xCord[ii]][yCord[ii]]=pixelVal
                     processMap[xCord[ii]][yCord[ii]]=False
                     UnprocessedPixels=UnprocessedPixels-1 #decrease the counter of pixels still to be processed
                SumVal=0
                numberofUneliminatedNeighbors=0
    return reconstructIris






##################For Testing###########################
if __name__ == '__main__':
#plt.ion()#uncomment if you don't vant to plot stuff
    F=plt.imread('/Users/Marike/Documents/MATLAB/iriscode/diagnostics/0002right_7-polar.jpg')
    HistFrac=0.1
    RecVal=40

#plt.imshow(F,cmap="gray") #Uncomment to show initial image 
#plt.show()

#K=noiseremover(F,HistFrac,RecVal)
#plt.hist(K.ravel(),256,range=(0,256),density=False)
#plt.show()
#L=equalisehistogram(F,40)

#plt.imshow(L,cmap="gray") #Uncomment to show final image 
#plt.show()


