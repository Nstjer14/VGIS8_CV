import matplotlib.pyplot as plt
import cv2

def imagePatchGenerator5(img,plotPatches = False):
    
    ratio = 64/70
    input_dim = 64
    patchDim = int(ratio*input_dim)
    margin = input_dim-patchDim    
    centMargin = margin//2
    
    patchCenter = img[centMargin:patchDim+centMargin,centMargin:patchDim+centMargin] 
    patchTopLeft = img[0:patchDim,0:patchDim]    
    patchTopRight = img[margin:patchDim+margin,0:patchDim]    
    patchBottomLeft = img[0:patchDim,margin:patchDim+margin]    
    patchBottomRight = img[margin:patchDim+margin,margin:patchDim+margin]
    if plotPatches == True:
        plt.subplot(4, 2, 1)
        #plt.imshow(img[:,:,0], cmap='gray')
        plt.imshow(img, cmap='gray')
        plt.gca().set_title('Original')
        plt.gca().set_xlim([0,input_dim])
        plt.gca().set_ylim([0,input_dim])
        print("Original shape",img.shape)
        
        plt.subplot(3, 2, 2)
        plt.gca().set_title('Center Right')
        #plt.imshow(patchCenter[:,:,0], cmap='gray')
        plt.imshow(patchCenter, cmap='gray')
        plt.gca().set_xlim([0,input_dim])
        plt.gca().set_ylim([0,input_dim])
        print("Center",patchCenter.shape)
        
        plt.subplot(3, 2, 3)
        plt.gca().set_title('Top Left')
        #plt.imshow(patchTopLeft[:,:,0], cmap='gray')
        plt.imshow(patchTopLeft, cmap='gray')
        plt.gca().set_xlim([0,input_dim])
        plt.gca().set_ylim([0,input_dim])
        print("Top left",patchTopLeft.shape)
        
        plt.subplot(3, 2, 4)
        plt.gca().set_title('Top Right')
        #plt.imshow(patchTopRight[:,:,0], cmap='gray')
        plt.imshow(patchTopRight, cmap='gray')
        plt.gca().set_xlim([0,input_dim])
        plt.gca().set_ylim([0,input_dim])
        print("Top Right",patchTopRight.shape)
        
        plt.subplot(3, 2, 5)
        plt.gca().set_title('Bottom Left')
        #plt.imshow(patchBottomLeftt[:,:,0], cmap='gray')
        plt.imshow(patchBottomLeft, cmap='gray')
        plt.gca().set_xlim([0,input_dim])
        plt.gca().set_ylim([0,input_dim])
        print("Bottom Left",patchBottomLeft.shape)
    
        plt.subplot(3, 2, 6)
        plt.gca().set_title('Bottom Right')
        #plt.imshow(patchBottomRight[:,:,0], cmap='gray')
        plt.imshow(patchBottomRight, cmap='gray')
        plt.gca().set_xlim([0,input_dim])
        plt.gca().set_ylim([0,input_dim])
        print("Bottom Right",patchBottomRight.shape)
       
        currentPlot = plt.gcf()
        #currentPlot.tight_layout()
       
    return (patchCenter, patchTopLeft, patchTopRight, patchBottomLeft, patchBottomRight)