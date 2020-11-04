import utm
import cv2
import numpy as np



def paste_image(new_img, old_img, new_img_center_lat, new_img_center_long, old_img_center_x: 'utm coords', old_img_center_y: 'utm coords'):
    """
    new_img: image you want to add to old image
    old_img: one image that is a collection of images that were previously placed together
    """
    new_img_h, new_img_w, _ = new_img.shape
    new_img_center_x, new_img_center_y, _, _ = utm.from_latlon(new_img_center_lat, new_img_center_long)
    new_img_left = new_img_center_x - (new_img_w / 2)
    new_img_right = new_img_center_x + (new_img_w / 2)
    new_img_top = new_img_center_y + (new_img_h / 2)
    new_img_bottom = new_img_center_y - (new_img_h / 2)

    old_img_h, old_img_w, _, = old_img.shape
    old_img_left = old_img_center_x - (old_img_w / 2)
    old_img_right = old_img_center_x + (old_img_w / 2)
    old_img_top = old_img_center_y + (old_img_h / 2)
    old_img_bottom = old_img_center_y - (old_img_h / 2)    

    final_img_left = min(new_img_left, old_img_left)
    final_img_right = max(new_img_right, old_img_right)
    final_img_bottom = min(new_img_bottom, old_img_bottom)
    final_img_top = max(new_img_top, old_img_top)
    final_img_h = final_img_top - final_img_bottom
    final_img_w = final_img_right - final_img_left

    new_utm_center_x = final_img_left + (final_img_w / 2)
    new_utm_center_y = final_img_top - (final_img_h / 2)

    # move utm coords to positive origin
    final_img_right = final_img_right - final_img_left
    final_img_top -= final_img_bottom
    final_img_left = 0
    final_img_bottom = 0

    # find new coords of new and old images w.r.t. moved final img
    new_img_adj_left, new_img_adj_right, new_img_adj_top, new_img_adj_bottom = 0, 0, 0, 0
    old_img_adj_left, old_img_adj_right, old_img_adj_top, old_img_adj_bottom = 0, 0, 0, 0
    if new_img_left > old_img_left:
        new_img_adj_left = 0
        new_img_adj_right = new_img_w
        old_img_adj_left = final_img_right - old_img_w 
        old_img_adj_right = final_img_right
    else:
        new_img_adj_left = final_img_right - new_img_w 
        new_img_adj_right = final_img_right 
        old_img_adj_left = 0
        old_img_adj_right = old_img_w

    if new_img_bottom > old_img_bottom:
        new_img_adj_bottom = 0
        new_img_adj_top = new_img_h
        old_img_adj_bottom = final_img_top - old_img_h
        old_img_adj_top = final_img_top
    else:
        new_img_adj_bottom = final_img_top - new_img_h
        new_img_adj_top = final_img_top 
        old_img_adj_bottom = 0
        old_img_adj_top = old_img_h

    # place two images on final image
    final_img = np.zeros((int(final_img_h), int(final_img_w), 3), np.uint8)
    final_img[int(new_img_adj_bottom):int(new_img_adj_top), int(new_img_adj_left):int(new_img_adj_right),:3] = new_img
    final_img[int(old_img_adj_bottom):int(old_img_adj_top), int(old_img_adj_left):int(old_img_adj_right),:3] = old_img

    cv2.imshow('Final Image', final_img)
    cv2.waitKey(0)

    return new_utm_center_x, new_utm_center_y 


if __name__=='__main__':
    # skeleton of what code needs
    latlon_dict = {'new_image': (lat, lon)}
    utm_center_coords = {'old_image': (x, y)}
    new_img = cv2.imread(new_img_path)
    old_img = cv2.imread(old_img_path)
    paste_image(new_img, old_img, latlon_dict['new_image'][0], latlon_dict['new_image'][1], utm_center_coords['old_image'][0], utm_center_coords['old_image'][1])
