#refer to corresponding script file in reference
import os
import numpy as np
import scipy as sp

class Bbox(): __slots__ = 'x1 y1 x2 y2'

def make_cub_anno_files(config):
    #correct them annotation files for the CUB 2011
    print("Making cub annotation files...\n")
    CUB_DIR = config.cub_root;

    [out_dir, fname, ext] = fileparts(config.train_image_fname)
    if not os.path.isdir(out_dir):
        print('Already done!')
        return

    #Keypoints, bounding boxes, paths, classes, split for every images
    im_part_fname = os.path.join(CUB_DIR, 'parts', 'part_locs.txt')
    im_bbox_fname = os.path.join(CUB_DIR, 'bounding_boxes.txt')
    im_path_fname = os.path.join(CUB_DIR, 'images.txt')
    im_split_fname = os.path.join(CUB_DIR, 'train_test_split.txt')
    im_label_fname = os.path.join(CUB_DIR, 'image_class_labels.txt')

    path_data =
    im_ids = path_data[0, :]
    rel_paths = path_data[1, :]
    images = []
    for i in range(len(im_ids))
        image_id = im_ids[i]
        rel_path = rel_paths{i}
        images[image_id] = {}
        images[image_id]['id'] = image_id
        images[image_id]['rel_path'] = rel_path

    #image sizes
    all_ims = np.empty((len(im_ids), ))
    for i in range(len(im_ids)):
        if i % 256 == 0:
            print('im size {}/{}'.format(i, len(im_ids))))
        im_path = os.path.join(CUB_DIR, 'images', images[i]['rel_path'])
        im = sp.imread(im_path)
        if np.size(im, axis = 2) == 1:
            im = np.stack([im, im, im], axis = 2)
        all_ims[i] = im
        images[i]['width'] = np.size(im, axis = 1)
        images[i]['height'] = np.size(im, axis = 0)

    #Splits
    split_data = #figure out input
    im_ids = split_data[0, :]
    is_training = split_data[1, :]
    for i in range(len(im_ids)):
        images[im_ids[i]]['train'] = is_training[i] == 1
        images[im_ids[i]]['test'] = is_training[i] == 0

    #classes
    class_data = #figure out input
    im_ids = class_data[0, :]
    labels = class_data[1, :]
    for i in range(len(im_ids)):
        images[im_ids[i]]['class'] = labels[i]

    #bounding_boxes
    bbox_data = #figure out input
    im_ids = bbox_data[0, :]
    xs = bbox_data[1, :]
    ys = bbox_data[2, :]
    widths = bbox_data[3, :]
    heights = bbox_data[4, :]
    for i in range(len(im_ids)):
        bbx = Bbox()
        bbx.x1 = int(xs[i])
        bbx.y1 = int(ys[i])
        bbx.x2 = int(x1 + widths[i] - 1)
        bbx.y2 = int(y1 + heights[i] - 1)
        images[im_ids]['bbox'] = bbx

    im_save_fname = config.image_fname
    #maybe use some pickling

    all_images = images

    #Make testing, training fileparts
    print('Saving ...')
    images = [i for i in all_images if i['test'] == True]
    #pickle this thing into a file
    images = [i for i in all_images if i['train'] == True]
    #pickle this thing into a file

    #Save files including images
    images = [i for i in all_images if i['test'] == True]
    ims = [all_ims[i] for i in range(len(all_images)) if all_images[i]['test'] == True]
    #save this baby

    images = [i for i in all_images if i['train'] == True]
    ims = [all_ims[i] for i in range(len(all_images)) if all_images[i]['train'] == True]
    #save this baby
    print('done!')
