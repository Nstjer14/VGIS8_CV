import matplotlib.pyplot as plt
import cv2

def imageBatchGenerator5(img,plotBatches = False):
    
    ratio = 64/70
    input_dim = 64
    batchDim = int(ratio*input_dim)
    margin = input_dim-batchDim    
    centMargin = margin//2
    
    batchCenter = img[centMargin:batchDim+centMargin,centMargin:batchDim+centMargin] 
    batchTopLeft = img[0:batchDim,0:batchDim]    
    batchTopRight = img[margin:batchDim+margin,0:batchDim]    
    batchBottomLeft = img[0:batchDim,margin:batchDim+margin]    
    batchBottomRight = img[margin:batchDim+margin,margin:batchDim+margin]
    if plotBatches == True:
        plt.subplot(4, 2, 1)
        plt.imshow(img[:,:,0], cmap='gray')
        plt.gca().set_title('Original')
        print("Original shape",img.shape)
        
        plt.subplot(4, 2, 2)
        plt.gca().set_title('Center Right')
        plt.imshow(batchCenter[:,:,0], cmap='gray')
        plt.gca().set_xlim([0,input_dim])
        plt.gca().set_ylim([0,input_dim])
        print("Center",batchCenter.shape)
        
        plt.subplot(4, 2, 3)
        plt.gca().set_title('Top Left')
        plt.imshow(batchTopLeft[:,:,0], cmap='gray')
        plt.gca().set_xlim([0,input_dim])
        plt.gca().set_ylim([0,input_dim])
        print("Top left",batchTopLeft.shape)
        
        plt.subplot(4, 2, 4)
        plt.gca().set_title('Top Right')
        plt.imshow(batchTopRight[:,:,0], cmap='gray')
        plt.gca().set_xlim([0,input_dim])
        plt.gca().set_ylim([0,input_dim])
        print("Top Right",batchTopRight.shape)
        
        plt.subplot(4, 2, 5)
        plt.gca().set_title('Bottom Left')
        plt.imshow(batchTopRight[:,:,0], cmap='gray')
        plt.gca().set_xlim([0,input_dim])
        plt.gca().set_ylim([0,input_dim])
        print("Bottom Left",batchBottomLeft.shape)
    
        plt.subplot(4, 2, 6)
        plt.gca().set_title('Bottom Right')
        plt.imshow(batchTopRight[:,:,0], cmap='gray')
        plt.gca().set_xlim([0,input_dim])
        plt.gca().set_ylim([0,input_dim])
        print("Bottom Right",batchBottomRight.shape)
       
        currentPlot = plt.gcf()
        currentPlot.tight_layout()
       
    return (batchCenter, batchTopLeft, batchTopRight, batchBottomLeft, batchBottomRight)