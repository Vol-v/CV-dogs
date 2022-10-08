

import xml.etree.ElementTree as ET
import glob
import os
import json
import cv2




classes = []
input_dir = "./reid dataset raw/"
images_output_dir = "./reid dataset split/images_all/"

# create the image folder (output directory)
# os.mkdir(images_output_dir)



name_id = 0 # since I'm saving all bboxes from all frames of all vids I need to
all_vids = os.listdir("./reid dataset raw") #dirs of all vids
all_vids.remove('.ipynb_checkpoints')
print(all_vids)
# identify all the xml files in the annotations folder (input directory)
result = []
for directory in all_vids:
    files = glob.glob(os.path.join(input_dir,directory,"Annotations", '*.xml'))
    # loop through each 
    for fil in files:
        basename = os.path.basename(fil) #name of file in folder
        filename = os.path.splitext(basename)[0]
        # check if the label contains the corresponding image file
        if not os.path.exists(os.path.join(input_dir,directory,"JPEGImages", f"{filename}.PNG")):
            print(f"{filename} image does not exist!")
            continue

        # parse the content of the xml file
        tree = ET.parse(fil)
        root = tree.getroot()
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)

        for obj in root.findall('object'):
            label = obj.find("name").text
            # check for new classes and append to list
            if label not in classes:
                classes.append(label)  
            index = classes.index(label)
            pil_bbox = [round(float(x.text)) for x in obj.find("bndbox")]
            img = cv2.imread(os.path.join(input_dir,directory,"JPEGImages", f"{filename}.PNG"))
            crop_img = img[pil_bbox[1]:pil_bbox[3], pil_bbox[0]:pil_bbox[2]] # slice [y_min:y_max,x_min:x_max]
            new_imname = os.path.join(images_output_dir,f"img_{name_id}_{index}.PNG")
            result.append([str(new_imname),str(index),"0"]) #  [img_path,dog_id,cam_id]. only one cam so same id
            cv2.imwrite(new_imname,crop_img)
            name_id += 1
    print(f"directory {directory} completed!")
            
with open(os.path.join("reid dataset split","labels.txt"),"w",encoding="utf-8") as f:
    for r in result:
        f.write(" ".join(r)+"\n") #write to lbl_idx.txt tuple (img_path,dog_id,cam_id). only one cam so same id

#generate the classes file as reference
with open(os.path.join("reid dataset split","classes.txt"), 'w', encoding='utf8') as f:
    f.write(json.dumps(classes))