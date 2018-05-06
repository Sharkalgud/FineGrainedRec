import numpy as np
import scipy as sp
import time as time

def coseg_main(params):
    coseg_save_dir = params['im_out_dir']

    im_data = open(params['im_meta_fname']).read() #needs to be tweaked based on what is in the file
    images = im_data['images'] #needs modification similar to above line
    classes = list(set(images['classes'])) #broken possibility of using numpy to making operations easier
    #options should be modified to work with new wrapper
    options = {}
    options['num_iters'] = params['coseg_iters']
    options['class_weight'] = params['class_weight']
    options['use_class_fg'] = params['use_class_fg']
    options['use_class_bg'] = params['use_class_bg']
    options['fg_prior'] = params['fg_prior']
    options['do_refine'] = params['do_refine']

    class_cosegs =  np.empty((1, len(classes)))

    for class_ind in range(len(classes)):
        test_class = classes[class_ind]
        print('Coseg class {}\n'.format(test_class))

        #Load images and set them up for coseg_main
        stime = time.time()
        im_inds = find([images.class] == test_class);#replace with python once input functionalty figured out
        class_ims = np.empty((1, len(im_inds)))
        class_gt_masks = np.empty((1, len(im_inds)))
        class_min_fg_areas = np.zeros((1, len(im_inds)))
        class_max_fg_areas = np.ones((1, len(im_inds))) * 1000000
        class_min_fg_lengths = np.zeros((1, len(im_inds)))
        class_min_fg_heights = np.zeros((1, len(im_inds)))
        for i in range(len(im_inds))
            ind_str = str(im_inds[i])

            #get image
            im = open(params['im_base'] + images[im_inds[i]].rel_path) #broken fix when file input is fixed
            if np.size(im, 2) == 1:
                im = np.stack([im, im, im], axis = 2)
            scale = (params['resize_area'] / (np.size(im, axis = 1) * np.size(im, axis = 2))) ** (0.5)
            im_scaled = sp.misc.resize(im, scale)
            class_ims[i] = im_scaled #might be broken?

            #set up GT mask using the bounding box
            class_gt_masks[i] = 
