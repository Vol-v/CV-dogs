

import xml.etree.ElementTree as ET
import glob
import os
import json
import cv2


def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]


classes = []
input_dir = "../my_dogs_yolo/dataset-far/"
images_output_dir = "./reid dataset far/images/"

# create the image folder (output directory)
# os.mkdir(images_output_dir)



name_id = 0 # since I'm saving all bboxes from all frames of all vids I need to
all_labels = os.listdir(os.path.join(input_dir,"labels")) #dirs of all vids
# files = glob.glob(os.path.join(all_labels, '*.txt'))

# identify all the xml files in the annotations folder (input directory)
result = []
for fil in all_labels:
    filename = fil[:-4]
    # check if the label contains the corresponding image file
    if not os.path.exists(os.path.join(input_dir,"images", f"{filename}.PNG")):
        print(f"{filename} image does not exist!")
        continue
    boxes = []
    with open(os.path.join(input_dir,"labels",fil),"r",encoding="utf-8") as f:
        for string in f:
            if string != "":
                boxes.append(string)
    for box in boxes:
        data = box.strip().split(' ')
        bbox = [float(x) for x in data[1:]]
        img = cv2.imread(os.path.join(input_dir,"images", f"{filename}.PNG"))
        pil_bbox = yolo_to_xml_bbox(bbox, img.shape[1], img.shape[0])
        class_value = box[0]
        crop_img = img[pil_bbox[1]:pil_bbox[3], pil_bbox[0]:pil_bbox[2]]
        new_imname = os.path.join(images_output_dir, filename+"_"+class_value+".PNG")
        result.append([str(new_imname),class_value,"0"])
        cv2.imwrite(new_imname,crop_img)
    

with open(os.path.join("reid dataset far","labels.txt"),"w",encoding="utf-8") as f:
    for r in result:
        f.write(" ".join(r)+"\n") #write to lbl_idx.txt tuple (img_path,dog_id,cam_id). only one cam so same id