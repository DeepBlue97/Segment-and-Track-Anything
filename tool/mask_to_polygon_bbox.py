import os
import json

import cv2
import numpy as np


def findValueContours(img_gray, value):
    img_gray_thresh = np.zeros_like(img_gray)
    img_gray_thresh[img_gray == value] = 255  

    img_gray_contours, img_gray_hierarchy = cv2.findContours(img_gray_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return img_gray_contours, img_gray_hierarchy


if __name__ == "__main__":

    # src_imgs_folder = "assets/pallet_masks"
    src_imgs_folder = "tracking_results/pallet_short/pallet_short_masks"
    dst_json_path = 'coco.json'
    

    category_and_instance = {
        "pallet": {
            "id": 0,
            "supercategory": 'pallet',
            "instance": [
                {
                    "start_frame": 0,
                    "pix_value": 1,
                }
            ]
        }
    }


    cats = []
    imgs = []
    anns = []

    for catKey, catValue in category_and_instance.items():
        cats.append({
            "supercategory": catValue['supercategory'],
            "id": len(cats),
            "name": catKey,
        })
    
    img_filenames = os.listdir(src_imgs_folder)
    # img_gray_filenames = sorted([i for i in img_filenames if i.endswith('_gray.png')], key=lambda x: int(x.split('_')[0]))
    img_gray_filenames = sorted([i for i in img_filenames if i.startswith('gray_')], key=lambda x: int(x.split('_')[1].split('.')[0])) # parse gray_xxx.png format file


    for i, img_filename in enumerate(img_gray_filenames):
        img_id = int(img_filename.split('_')[1].split(".")[0])


        img_gray_fullname = os.path.join(src_imgs_folder, img_filename)
        img_ori_fullname = os.path.join(src_imgs_folder, "ori_"+str(img_id).zfill(5)+".jpg")
        img_contour_fullname = os.path.join(src_imgs_folder, "contour_"+str(img_id).zfill(5)+".jpg")

        img_gray = cv2.imread(img_gray_fullname, cv2.IMREAD_GRAYSCALE)
        img_ori = cv2.imread(img_ori_fullname)

        imgs.append({
            "file_name": os.path.basename(img_ori_fullname),
            "width": img_ori.shape[1],
            "height": img_ori.shape[0],
            "id": img_id
        })

        # img_gray_ret, img_gray_thresh = cv2.threshold(img_gray, 11, 255, cv2.THRESH_BINARY)
        # img_gray_ret, img_gray_thresh = cv2.threshold(img_gray, 12, 0, cv2.THRESH_TOZERO)

        # img_gray_thresh = np.zeros_like(img_gray)
        # img_gray_thresh[img_gray == 12] = 255  
        # img_gray_contours, img_gray_hierarchy = cv2.findContours(img_gray_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for catKey, catValue in category_and_instance.items():
            # if catValue[]
            for ins in catValue['instance']:
                if ins['start_frame'] <= i:
                    img_gray_contours, img_gray_hierarchy = findValueContours(img_gray, ins["pix_value"])

                    # assert len(img_gray_contours) == 1
                    contours_points_count = [j.size for j in img_gray_contours]
                    target_contour_index = contours_points_count.index(max(contours_points_count))
                    x, y, w, h = cv2.boundingRect(img_gray_contours[target_contour_index])  # 仅bbox轮廓点最多的轮廓
                    cv2.rectangle(img_ori, (x, y), (x+w, y+h), (255, 0, 0), 1)  # 用蓝色绘制矩形

                    cv2.drawContours(img_ori, img_gray_contours, -1, (0, 255, 0), 1)
                    cv2.imwrite(img_contour_fullname, img_ori)

                    anns.append({
                        "segmentation": [int(k) for k in img_gray_contours[target_contour_index].squeeze().flatten()],
                        "area": cv2.contourArea(img_gray_contours[target_contour_index]),
                        "iscrowd": 0,
                        "image_id": img_id,
                        "bbox": [
                            x, y, w, h
                        ],
                        "category_id": [k for k in cats if k["name"]==catKey][0]['id'],
                        "id": len(anns)
                    })

        
        print(i+1, len(img_gray_filenames))

    with open(dst_json_path, 'w') as f:
        
        json.dump({
            "images": imgs,
            "annotations": anns,
            "categories": cats
        }, f)
