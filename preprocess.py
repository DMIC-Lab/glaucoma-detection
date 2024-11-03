# %%
import numpy as np
import os
import cv2
from skimage.filters import frangi, gaussian
import skimage.morphology as morphology
from scipy.ndimage import zoom

# %%
imagePaths = sorted([os.path.join('train',file) for file in os.listdir("train") if '.JPG' in file],key=lambda path: path.lower())
imagePaths = imagePaths[len(imagePaths)//2:]

# %%
def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

# %%
import cupy as cp
from cupyx.scipy.ndimage import zoom as cpzoom, gaussian_filter as cpgaussian
from cupyx.scipy.ndimage import binary_dilation as cpdilation

# %%
def process_image_cupy(image):
    # Convert image to CuPy array
    image_cp = cp.asarray(image)
    originalImage_cp = image_cp.copy()
    originalShape = image.shape

    # Resize image to 512x512
    image_cp = cpzoom(image_cp, (512 / image.shape[0], 512 / image.shape[1], 1))
    mask_cp = cp.zeros_like(image_cp)
    mask_cp[50:-50, 20:-20] = True
    image_cp = cp.where(mask_cp, image_cp, 0)

    blackMask_cp = cp.all(image_cp < 5, axis=-1)

    # Convert to grayscale (requires CPU computation)
    image_np = cp.asnumpy(image_cp)
    image_gray_np = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    image_gray_cp = cp.asarray(image_gray_np)

    for _ in range(30):
        blackMask_cp = cpdilation(blackMask_cp)
    image_cp = cpgaussian(image_gray_cp, sigma=3)
    grad = cp.array(cp.gradient(image_cp))
    grad_cp = cp.linalg.norm(grad, axis=0)
    image_cp = cp.where(blackMask_cp, 0, image_cp)
    grad_cp = cp.where(blackMask_cp, 0, grad_cp)

    # Find min and max locations (requires CPU computation)
    grad_image_np = cp.asnumpy(grad_cp * image_cp)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(grad_image_np)
    scaled_loc_x = max_loc[0] * (originalShape[1] / 512)
    scaled_loc_y = max_loc[1] * (originalShape[0] / 512)
    newBound_height = 0.125 * originalShape[0]
    newBound_width = 0.125 * originalShape[1]

    xmin = max(0, int(scaled_loc_x - newBound_width))
    xmax = min(originalShape[1] - 1, int(scaled_loc_x + newBound_width))
    ymin = max(0, int(scaled_loc_y - newBound_height))
    ymax = min(originalShape[0] - 1, int(scaled_loc_y + newBound_height))
    crop_cp = originalImage_cp[ymin:ymax, xmin:xmax]

    # Convert to grayscale (requires CPU computation)
    crop_np = cp.asnumpy(crop_cp)
    crop_gray_np = cv2.cvtColor(crop_np.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    distribution = np.sum(crop_gray_np, axis=0) / 1000
    diff = np.sum(distribution[:len(distribution) // 2]) - np.sum(distribution[len(distribution) // 2:])
    #print(diff, np.sum(distribution[:len(distribution) // 2]),
    #      diff / np.sum(distribution[:len(distribution) // 2]))
    if (np.abs(diff) / np.sum(distribution[:len(distribution) // 2])) > 0.3:
        if diff > 0:
            scaled_loc_x -= 0.2 * len(distribution)
        else:
            scaled_loc_x += 0.2 * len(distribution)
    xmin = max(0, int(scaled_loc_x - newBound_width))
    xmax = min(originalShape[1] - 1, int(scaled_loc_x + newBound_width))
    ymin = max(0, int(scaled_loc_y - newBound_height))
    ymax = min(originalShape[0] - 1, int(scaled_loc_y + newBound_height))
    crop_cp = originalImage_cp[ymin:ymax, xmin:xmax]
    final_cp = cpzoom(crop_cp, (512 / crop_cp.shape[0], 512 / crop_cp.shape[1], 1))
    final_np = cp.asnumpy(final_cp)
    # Apply CLAHE
    # Convert to LAB color space
    lab = cv2.cvtColor(final_np, cv2.COLOR_BGR2LAB)
    # Split the LAB image into its channels
    l_channel, a_channel, b_channel = cv2.split(lab)
    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_channel)
    # Merge the CLAHE enhanced L-channel back with a and b channels
    lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
    # Convert back to BGR color space
    final_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return final_clahe

# %%

# def process_image(image):
#     originalImage = image.copy()
#     originalShape = image.shape
#     # Resize image to 512x512
#     image = zoom(image, (512 / image.shape[0], 512 / image.shape[1], 1))
#     mask = np.zeros_like(image)
#     mask[50:-50, 20:-20] = True
#     image = np.where(mask, image, 0)

#     blackMask = np.all(image < 5, axis=-1)

#     image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)

#     for _ in range(30):
#         blackMask = morphology.binary_dilation(blackMask)
#     image = gaussian(image, sigma=3)
#     grad = np.linalg.norm(np.gradient(image), axis=0)
#     image = np.where(blackMask, 0, image)
#     grad = np.where(blackMask, 0, grad)

#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(grad * image)
#     # max_loc is (x, y)
#     scaled_loc_x = max_loc[0] * (originalShape[1] / 512)
#     scaled_loc_y = max_loc[1] * (originalShape[0] / 512)
#     newBound_height = 0.125 * originalShape[0]
#     newBound_width = 0.125 * originalShape[1]

#     xmin = max(0, int(scaled_loc_x - newBound_width))
#     xmax = min(originalShape[1] - 1, int(scaled_loc_x + newBound_width))
#     ymin = max(0, int(scaled_loc_y - newBound_height))
#     ymax = min(originalShape[0] - 1, int(scaled_loc_y + newBound_height))
#     crop = originalImage[ymin:ymax, xmin:xmax]

#     distribution = np.sum(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), axis=0) / 1000
#     diff = np.sum(distribution[:len(distribution) // 2]) - np.sum(distribution[len(distribution) // 2:])
#     #print(diff, np.sum(distribution[:len(distribution) // 2]),
#           #diff / np.sum(distribution[:len(distribution) // 2]))
#     if (np.abs(diff) / np.sum(distribution[:len(distribution) // 2])) > 0.3:
#         if diff > 0:
#             scaled_loc_x -= 0.2 * len(distribution)
#         else:
#             scaled_loc_x += 0.2 * len(distribution)
#     xmin = max(0, int(scaled_loc_x - newBound_width))
#     xmax = min(originalShape[1] - 1, int(scaled_loc_x + newBound_width))
#     ymin = max(0, int(scaled_loc_y - newBound_height))
#     ymax = min(originalShape[0] - 1, int(scaled_loc_y + newBound_height))
#     crop = originalImage[ymin:ymax, xmin:xmax]
#     final = zoom(crop, (512 / crop.shape[0], 512 / crop.shape[1], 1))
#     return final

# %%
testImageBatches = [cv2.imread(imagePath) for imagePath in imagePaths[:50]]

# %%
l = len(imagePaths)
os.makedirs('preprocessed',exist_ok=True)
for i,path in enumerate(imagePaths):
    image = cv2.imread(path)
    result = process_image_cupy(image)
    if i % 10 == 0:
        print(i,l,end='\r')
    # if i < 10:
    #     plt.imshow(result)
    #     plt.show()
    # else:
    #     break
    np.save(os.path.join('preprocessed',os.path.basename(path).replace('.JPG','.npy')),result)

# %%
# for i in range(50):
#     imagePath = imagePaths[i]
#     image = cv2.imread(imagePath)
#     originalImage = image.copy()
#     originalShape = image.shape
#     image = zoom(image,(512/image.shape[0],512/image.shape[1],1))
    
#     mask = np.zeros_like(image)
#     mask[50:-50,20:-20] = True
#     image = np.where(mask,image,0)

#     blackMask = np.all(image < 5, axis=-1)

#     image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
#     for _ in range(30):
#         blackMask = morphology.binary_dilation(blackMask)
#     image = gaussian(image,sigma=3)
#     grad = np.linalg.norm(np.gradient(image),axis=0)
#     image = np.where(blackMask,0,image)
#     grad = np.where(blackMask,0,grad)

#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(grad*image)
#     scaled_loc = max_loc[0] * (originalShape[1] / 512), max_loc[1] * (originalShape[0] / 512)
#     newBound = [.125 * val for val in originalShape[:2]]
#     xmin = max(0,int(scaled_loc[1]-newBound[0]))
#     xmax = min(originalShape[1]-1,int(scaled_loc[1] + newBound[0]))
#     ymin = max(0,int(scaled_loc[0]-newBound[1]))
#     ymax = min(originalShape[0]-1,int(scaled_loc[0] + newBound[1]))
#     crop =originalImage[xmin:xmax,ymin:ymax]


#     distribution = np.sum(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),axis=0) / 1000
#     original_loc = scaled_loc
#     diff = np.sum(distribution[:len(distribution)//2]) - np.sum(distribution[len(distribution)//2:])
#     print(diff,np.sum(distribution[:len(distribution)//2]), diff / np.sum(distribution[:len(distribution)//2]))
#     if (np.abs(diff) / np.sum(distribution[:len(distribution)//2])) > .3:
#         if diff > 0:    
#             scaled_loc = scaled_loc[0]- .2 * len(distribution) , scaled_loc[1]
#         else:
#             scaled_loc  = scaled_loc[0] + .2* len(distribution), scaled_loc[1]
#     xmin = max(0,int(scaled_loc[1]-newBound[0]))
#     xmax = min(originalShape[1]-1,int(scaled_loc[1] + newBound[0]))
#     ymin = max(0,int(scaled_loc[0]-newBound[1]))
#     ymax = min(originalShape[0]-1,int(scaled_loc[0] + newBound[1]))
#     crop =originalImage[int(scaled_loc[1]-newBound[0]):int(scaled_loc[1] + newBound[0]),int(scaled_loc[0]-newBound[1]):int(scaled_loc[0] + newBound[1])]
#     final = zoom(crop,(512/crop.shape[0],512/crop.shape[1],1))

    

    



