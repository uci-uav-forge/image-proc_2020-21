import cv2
import time
import numpy as np
import math
import os
import itertools
from sys import stdout
from threading import Thread

BOX_SAVE_DIR = './boxes/'

def gen_horiz_kernel(size, vclip):
    '''
    Generate a horizontal-biased kernel

    Parameters
    ----------
    size: int
        the size of the kernel
    vclip: float, 0 to 1
        the fractional amount of vertical pixels to clip off top and bottom of the kernel.
        e.g. for a kernel of size 4:

        no clipping:        0.5 clipping:
        ####                0000               
        ####                ####
        ####                ####
        ####                0000
    
    Return
    ------
    np array of uint8
        The pixel values of the kernel
    '''


    if vclip > 1 or vclip < 0:
        raise ValueError("vclip must be between 0 and 1")
    kernel = np.ones((size,size), np.uint8)
    clip = round(vclip * size * .5)
    kernel[0:clip,:] = 0
    kernel[(size-clip):size] = 0
    return kernelt

def circle_kern(w, h):
    ''' 
    Compute a circular kernel of width w and height h.

    Parameters
    ----------
    w, h: int
        The width and height of the kernel.
    
    Returns
    -------
    np array of uint8
        The pixel values of the kernel.
    '''
    center = (int(w/2), int(h/2))
    radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    kernel = dist_from_center <= radius
    return kernel.astype(np.uint8)

def color_thresh(hsv_img, blur_rad, clip_size, lower_h, upper_h, sat_thresh, val_thresh):
    '''
    Generate a color threshhold from hsv img
    '''
    blurred = cv2.blur(hsv_img, (blur_rad, blur_rad))  

    blurred = cv2.GaussianBlur(hsv_img, (blur_rad, blur_rad), 0)
    if lower_h > upper_h:
        min1 = np.array([0, sat_thresh, val_thresh], np.uint8)
        max1 = np.array([upper_h, 255, 255], np.uint8)
        min2 = np.array([lower_h, sat_thresh, val_thresh], np.uint8)
        max2 = np.array([180, 255, 255], np.uint8)
        color_thresh1 = cv2.inRange(hsv_img, min1, max1)
        color_thresh2 = cv2.inRange(hsv_img, min2, max2)
        color_thresh = cv2.bitwise_or(color_thresh1, color_thresh2)
    else:
        min1 = np.array([lower_h, sat_thresh, val_thresh], np.uint8)
        max1 = np.array([upper_h, 255, 255], np.uint8)
        color_thresh = cv2.inRange(hsv_img, min1, max1)
    
    noise_kernel = circle_kern(1.2 * clip_size, clip_size)

    color_thresh = cv2.morphologyEx(color_thresh, cv2.MORPH_OPEN, noise_kernel)
    color_thresh = cv2.morphologyEx(color_thresh, cv2.MORPH_DILATE, noise_kernel)
    return color_thresh

def iou(A, B):
    ''' 
    Compute the intersection-over-union of two boxes A and B.

    Parameters
    ----------
    A, B: list of int
        The corners of the box in list form. [x1, y1, x2, y2]
    
    Returns
    -------
    float
        0.0 - 1.0 if boxes partially overlap
        1.0 if boxes overlap perfectly
        0.0 if boxes do not overlap
    '''

    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])
    intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    A_area = (A[2] - A[0] + 1) * (A[3] - A[1] + 1)
    B_area = (B[2] - B[0] + 1) * (B[3] - B[1] + 1)
    return intersection / float(A_area + B_area - intersection)

def smaller_box(A, B):
    ''' 
    Parameters
    ----------
    A, B: list of int
        The corners of the box in list form. [x1, y1, x2, y2]
    
    Returns
    -------
    bool
        True if A is smaller than B
        False if A is larger than B
    '''
    Aw = A[2] - A[0]
    Bw = B[2] - B[0]
    Ah = A[3] - A[1]
    Bh = B[3] - B[1]
    areaA = Aw * Ah
    areaB = Bw * Bh
    if areaA > areaB:
        return False
    if areaB > areaA:
        return True

def gen_hues(hi, hi2, lo):
    '''
    Generate an array of hue clipping values. All params are the 
    lower bound for the saturation threshhold.

    Parameters
    ----------
    hi: int
        High saturation clip value
    hi2: int
        Alternate high saturation clip value
    lo: int
        Low saturation clip value
    
    Returns
    -------
    dict of str : (tuple of int)
        The hues. The key is the string color name, the value is a 
        tuple with the lower hue bound, the upper hue bound, the 
        lower saturation bound, and the upper saturation bound.
    '''

    hues_raw = [
        ('red',      0,  lo,    100 ),
        ('orange',  30,  hi,    100 ),
        ('yellow',  60,  hi,    100 ),
        ('yelgrn',  90,  lo,    100 ),
        ('green',  120,  lo,    100 ),
        ('seafm',  150,  lo,    100 ),
        ('cyan',   180,  lo,    100 ),
        ('skyblu', 210,  lo,    100 ),
        ('blue',   240,  lo,    100 ),
        ('violet', 270,  lo,    100 ),
        ('magent', 300,  lo,    100 ),
        ('redmgt', 330,  lo,    100 ),]
    hues = {}
    for hue in hues_raw:
        # partition hues
        hues[hue[0]] = (int(hue[1]/2 - 7), # lower hue bound 
                        int(hue[1]/2 + 8), # upper hue bound
                        int(hue[2] - 5 * 2.55), # lower saturation bound, upper = 255
                        int(hue[3] - 5 * 2.55)) # lower value bound, upper = 255
    return hues


def colorblock(image, frame_no, blur_size, clip_size, save_boxes=True):
    '''
    Segment an image into its saturated color regions

    Parameters
    ----------
    image: np array of uint32
        The decoded video frame or image, BGR color.
    frame_no: int
        Which video frame
    blur_size: int
        The size to blur the image before segmentation
    clip_size: int
        The minimum size for segmented areas
    save_boxes: bool
        Whether to save rectangular images of segmented areas to disk
        If True, save image
        If False, do not save image, only display
    
    Returns
    -------
    np array of uint32
        The video frame with overlaid contours, boxes, centroids, & color names.
    '''
    hues = gen_hues(75, 85, 45)

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_gsc = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h,w = image.shape[:2]

    # the layer for visualizations
    vis = np.zeros((h, w), np.uint8)
    
    for hname, hue in hues.items():
        blank = np.zeros((h, w), np.uint8)
        threshd = color_thresh(img_hsv, blur_size, clip_size, hue[0], hue[1], hue[2], hue[3])
        ########## CONTOURS ##########
        contours, hierarchy = cv2.findContours(threshd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.convexHull(c) for c in contours if cv2.contourArea(c) > clip_size * clip_size * 6]
        cv2.drawContours(blank, contours, -1, (255, 255, 255), thickness=2)

        ########### BOXES ###########
        # fill list of boxes
        boxes = []
        for ic, c in enumerate(contours):
            box_x, box_y, box_w, box_h = cv2.boundingRect(c)
            boxes.append([box_x, box_y, box_x + box_w, box_y + box_h])
        
        # prune the smaller of 2 boxes that overlap
        lg_box_list = []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if not smaller_box(boxes[i], boxes[j]):
                    lg_box_list.append(i)
        boxes = [boxes[b] for b in lg_box_list]
        
        # draw boxes on overlay
        for box in boxes:
            boxleft = box[0]
            boxtop = box[1]
            boxright = box[2]
            boxbottom = box[3]
            cv2.rectangle(blank, (boxleft, boxtop), (boxright, boxbottom), (255, 0, 0), 1)
        
        # Save images of boxes
        for (x1, y1, x2, y2) in boxes:
            aspect = (x1-x2)/(y1-y2)
            # save approximately square images only
            if 0.85 < aspect and aspect < 1.6:
                extracted_target = image[y1:y2, x1:x2]
                if save_boxes:
                    fname = BOX_SAVE_DIR + str(hname) + '_' + str(frame_no) + '.jpg'
                    cv2.imwrite(fname, extracted_target)
                    print('Shape saved to {}'.format(fname))

        ########### MOMENTS, LABELS ###########
        moments = []
        for i, c in enumerate(contours):
            # compute moment x,y
            moment = cv2.moments(c)
            if moment['m00'] != 0:
                cX = int(moment["m10"] / moment["m00"])
                cY = int(moment["m01"] / moment["m00"])
            else:
                cX = 0
                cY = 0
            moments.append((cX, cY))
            cv2.circle(blank, (cX, cY), 3, (255, 255, 255))
            cv2.putText(blank, str(hname) + str(i), (cX - 20, cY - 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2)
        
        vis = cv2.bitwise_or(vis, blank)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_or(vis, image)

if __name__ == "__main__":
    # source, url or filename
    src = './data/test.mp4'

    # colorblock params (even int)
    blur_size = 51
    clip_size = 19

    # show the video or webcam (singlethreaded)
    show_singlethread = False
    # write the video or webcam stream to disk
    write = True

    if show_singlethread:
        video = cv2.VideoCapture(src)
        f=0
        while video.isOpened():
            ret, frame = video.read()
            frame = colorblock(frame, f, blur_size, clip_size, save_boxes=False)
            if ret == True:
                f+=1
                cv2.imshow('img', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        video.release()

    if write:
        video = cv2.VideoCapture(src)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        (w, h) = int(video.get(3)), int(video.get(4))
        fileout = './output.mp4'
        
        print('w={} h={}'.format(w, h))
        print('writing to {}'.format(fileout))

        video_writer = cv2.VideoWriter(fileout, fourcc, 30, (w, h))
        f = 0
        while video.isOpened():
            ret, frame = video.read()
            if ret == True:
                # display frame counter
                stdout.write("\tframe {} of {}{}".format(f, total_frames, "\r"))
                stdout.flush()
                f += 1
                frame = colorblock(frame, f, blur_size, clip_size, save_boxes=False)
                video_writer.write(frame)
            if ret == False:
                break
        video_writer.release()
        print('done! Wrote {} frames.'.format(f))
