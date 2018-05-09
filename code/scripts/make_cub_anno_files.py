#refer to corresponding script file in reference
import os
import numpy as np
import scipy as sp
import pickle

class Bbox(): __slots__ = 'x1 y1 x2 y2'

def make_cub_anno_files(config):
    #correct them annotation files for the CUB 2011
    print("Making cub annotation files...\n")
    CUB_DIR = config.cub_root;

    out_dir = os.path.dirname(config.train_image_fname)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    done_fname = os.path.join(out_dir, 'cub.done')
    if os.path.exists(out_dir):
        print('Already done!')
        return

    #Keypoints, bounding boxes, paths, classes, split for every images
    im_part_fname = os.path.join(CUB_DIR, 'parts', 'part_locs.txt')
    im_bbox_fname = os.path.join(CUB_DIR, 'bounding_boxes.txt')
    im_path_fname = os.path.join(CUB_DIR, 'images.txt')
    im_split_fname = os.path.join(CUB_DIR, 'train_test_split.txt')
    im_label_fname = os.path.join(CUB_DIR, 'image_class_labels.txt')

    path_data = open(im_path_fname, 'r').read().split('\n')[:5]
    path_data = [im.split(' ') for im in path_data]
    im_ids = [int(im[0]) for im in path_data]
    rel_paths = [str(im[1]) for im in path_data]
    images = []
    for i in range(len(im_ids)): #might half it depending on time
        image_id = im_ids[i]
        rel_path = rel_paths[i]
        images[image_id] = {}
        images[image_id]['id'] = image_id
        images[image_id]['rel_path'] = rel_path

    #image sizes
    all_ims = []
    for i in range(len(im_ids)):
        if i % 256 == 0:
            print('im size {}/{}'.format(i, len(im_ids)))
        im_path = os.path.join(CUB_DIR, 'images', images[i]['rel_path'])
        im = sp.imread(im_path)
        if np.size(im, axis = 2) == 1:
            im = np.stack([im, im, im], axis = 2)
        all_ims.append(im)
        images[i]['width'] = np.size(im, axis = 1)
        images[i]['height'] = np.size(im, axis = 0)
    #Splits
    split_data = open(im_split_fname, 'r').read().split('\n')[:-1]
    split_data = [im.split(' ') for im in split_data]
    im_ids = [int(im[0]) for im in split_data]
    is_training = [int(im[1]) for im in split_data]
    for i in range(len(im_ids)):
        images[im_ids[i]]['train'] = is_training[i] == 1
        images[im_ids[i]]['test'] = is_training[i] == 0

    #classes
    class_data = open(im_label_fname, 'r').read().split('\n')[:-1]
    class_data = [im.split(' ') for im in class_data]
    im_ids = [int(im[0]) for im in class_data]
    labels = [int(im[1]) for im in class_data]
    for i in range(len(im_ids)):
        images[im_ids[i]]['class'] = labels[i]

    #bounding_boxes
    bbox_data = open(im_bbox_fname, 'r').read().split('\n')[:-1]
    #possible chance to make this more efficient
    bbox_data = [im.split(' ') for im in bbox_data]
    im_ids = [int(im[0]) for im in bbox_data]
    xs = [float(im[1]) for im in bbox_data]
    ys = [float(im[2]) for im in bbox_data]
    widths = [float(im[3]) for im in bbox_data]
    heights = [float(im[4]) for im in bbox_data]
    for i in range(len(im_ids)):
        bbx = Bbox()
        bbx.x1 = int(xs[i])
        bbx.y1 = int(ys[i])
        bbx.x2 = int(x1 + widths[i] - 1)
        bbx.y2 = int(y1 + heights[i] - 1)
        images[im_ids]['bbox'] = bbx

    im_save_fname = config.image_fname
    with open(im_save_fname, 'wb') as f:
        pickle.dump(images, f)
        f.close()

    all_images = images

    #Make testing, training fileparts
    print('Saving ...')
    images = [i for i in all_images if i['test'] == True]
    with open(config.test_image_fname, 'wb') as f:
        pickle.dump(images, f)
        f.close()
    images = [i for i in all_images if i['train'] == True]
    with open(config.train_image_fname, 'wb') as f:
        pickle.dump(images, f)
        f.close()

    #Save files including images
    images = [i for i in all_images if i['test'] == True]
    ims = [all_ims[i] for i in range(len(all_images)) if all_images[i]['test'] == True]
    with open(config.test_imagedata_fname, 'wb') as f:
        pickle.dump(images, f)
        pickle.dump(ims, f)
        f.close()

    images = [i for i in all_images if i['train'] == True]
    ims = [all_ims[i] for i in range(len(all_images)) if all_images[i]['train'] == True]
    with open(config.train_imagedata_fname, 'wb') as f:
        pickle.dump(images, f)
        pickle.dump(ims, f)
        f.close()
    print('done!')
