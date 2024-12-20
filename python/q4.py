import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation


# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions    
    
    # Perform estimation of noise and then denoisue
    sigma_est = skimage.restoration.estimate_sigma(image, channel_axis=-1)
    denoised = skimage.restoration.denoise_wavelet(image, sigma=sigma_est, channel_axis=-1)
    
    gray = skimage.color.rgb2gray(denoised)

    # Find and apply threshold (Otsu method best?)
    thresh = skimage.filters.threshold_otsu(gray)
    bw = gray < thresh
    
    # Morphological operations (NOTE: Vary size)
    bw = skimage.morphology.closing(bw, skimage.morphology.square(3))
    bw = skimage.morphology.remove_small_objects(bw, min_size=30)
    bw = skimage.morphology.remove_small_holes(bw, area_threshold=30)
    
    # Label connected components
    labeled_image = skimage.measure.label(bw)
    
    # Find bounding boxes
    regions = skimage.measure.regionprops(labeled_image)
    for region in regions:
        if region.area >= 50:  # Filter out small boxes
            minr, minc, maxr, maxc = region.bbox
            bboxes.append((minr, minc, maxr, maxc))
    
    bw = bw.astype(float)
    return bboxes, bw