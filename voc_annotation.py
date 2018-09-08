import xml.etree.ElementTree as ET
from os import getcwd
import os
import glob

sets=[('obj_train'), ('obj_val'), ('no_obj_train'), ('no_obj_val')]

classes = ["obj", "no_obj"]


def convert_annotation(image_id):
    in_file = open('%s.xml'%(image_id).replace('JPEGImages', 'Annotations'))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        # list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        return " " + ",".join([str(a) for a in b]) + ',' + str(cls_id)

wd = getcwd()

image_ids = glob.glob('voc' + os.sep + 'JPEGImages' + os.sep + '*.*')
image_id_paths = [(image_id + \
        ' ' + convert_annotation(image_id.split('.')[0]) + '\n') for image_id in image_ids]
with open('voc/list_master.txt', 'w') as list_file:
    list_file.write(''.join(list(image_id_paths)))


