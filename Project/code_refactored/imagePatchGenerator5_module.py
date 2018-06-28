import matplotlib.pyplot as plt
import cv2

def imagePatchGenerator5(img,input_dim = 64,plotPatches = False):
    '''
    imagePatchGenerator5(img,plotPatches = False) takes an image and produces
    5 patches that are slighly smaller than the original. It creates an image
    for each corner and of the center.
    img: The image that the patches are created from
    input_dim: the input image dimensions. Assumes square image
    plotPatches: if False then the patches are not plotted.
    '''
    ratio = 64/70
    #input_dim = 64
    patchDim = int(ratio*input_dim)
    margin = input_dim-patchDim    
    centMargin = margin//2
    
    patchCenter = img[centMargin:patchDim+centMargin,centMargin:patchDim+centMargin] 
    patchTopLeft = img[0:patchDim,0:patchDim]    
    patchBottomLeft = img[margin:patchDim+margin,0:patchDim]    
    patchTopRight = img[0:patchDim,margin:patchDim+margin]    
    patchBottomRight = img[margin:patchDim+margin,margin:patchDim+margin]
    if plotPatches == True:
        plt.figure(figsize=[10,6])
        plt.subplot(2, 3, 1)
        #plt.imshow(img[:,:,0], cmap='gray')
        plt.imshow(img, cmap='gray')
        plt.gca().set_title('Original')
        plt.gca().set_xlim([0,input_dim])
        plt.gca().set_ylim([input_dim,0])
        print("Original shape",img.shape)
        
        plt.subplot(2, 3, 2)
        plt.gca().set_title('Center')
        plt.imshow(patchCenter, cmap='gray')
        plt.gca().set_xlim([0,input_dim])
        plt.gca().set_ylim([input_dim,0])
        print("Center",patchCenter.shape)
        
        plt.subplot(2, 3, 3)
        plt.gca().set_title('Top Left')
        plt.imshow(patchTopLeft, cmap='gray')
        plt.gca().set_xlim([0,input_dim])
        plt.gca().set_ylim([input_dim,0])
        print("Top left",patchTopLeft.shape)
        
        plt.subplot(2, 3, 4)
        plt.gca().set_title('Top Right')
        plt.imshow(patchTopRight, cmap='gray')
        plt.gca().set_xlim([0,input_dim])
        plt.gca().set_ylim([input_dim,0])
        print("Top Right",patchTopRight.shape)
        
        plt.subplot(2, 3, 5)
        plt.gca().set_title('Bottom Left')
        plt.imshow(patchBottomLeft, cmap='gray')
        plt.gca().set_xlim([0,input_dim])
        plt.gca().set_ylim([input_dim,0])
        print("Bottom Left",patchBottomLeft.shape)
    
        plt.subplot(2, 3, 6)
        plt.gca().set_title('Bottom Right')
        plt.imshow(patchBottomRight, cmap='gray')
        plt.gca().set_xlim([0,input_dim])
        plt.gca().set_ylim([input_dim,0])
        print("Bottom Right",patchBottomRight.shape)
       
        currentPlot = plt.gcf()
        #currentPlot.tight_layout()
        plt.show()
       
    return (patchCenter, patchTopLeft, patchTopRight, patchBottomLeft, patchBottomRight)

if __name__ == '__main__':
    pass