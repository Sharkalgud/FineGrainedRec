import numpy as np
import scipy as sp
import time as time

#might have to define python object to hold meta date for images
def coseg(params):
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
        class_ims = np.empty((1, len(im_inds))) #might have to change for a more manageable array
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
            class_ims[i] = im_scaled #yeah its broken, you can't throw an array into this

            #set up GT mask using the bounding box
            class_gt_masks[i] = 10 * np.ones((np.size(im_scaled, axis = 0), np.size(im_scaled, axis = 2))) #also broken like above
            #Bbox init
            bbox = images[im_inds[i]].bbox;
            x1 = max(1, int((bbox['x1']-1)*scale+1 - params['bbox_context']));
            x2 = min(size(im_scaled, 2), int((bbox['x2']-1)*scale+1 + params['bbox_context']));
            y1 = max(1, int((bbox['y1']-1)*scale+1 - params['bbox_context']));
            y2 = min(size(im_scaled, 1), round((bbox['y2']-1)*scale+1 + params['bbox_context']));

            #GC_BGD = 0, GC_FGD = 1, GC_PR_BGD = 2, GC_PR_FGD = 3
            class_gt_masks[i, :] = 0;
            class_gt_masks[i, y1:y2, x1:x2] = 3
            if 0 in class_gt_masks[i]:
                class_gt_masks[i, 1, :) = 0;
                class_gt_masks[i, :, 1) = 0;
                class_gt_masks[i, end, :) = 0;
                class_gt_masks[i, :, end) = 0;
            #check assert
            bbox_length = x2 - x1 + 1
            bbox_height = y2 - y1 + 1
            bbox_area = bbox_length * bbox_height
            class_min_fg_areas[i] = params['bbox_min_fg_area'] * bbox_area;
            class_max_fg_areas[i] = params['bbox_max_fg_area'] * bbox_area;
            class_min_fg_lengths[i] = params['bbox_min_fg_length'] * bbox_length;
            class_min_fg_heights[i] = params['bbox_min_fg_height'] * bbox_height;
        elapsed = time.time() - stime
        print('Loading time class {}: {} sec\n'.format(test_class, elapsed))

        stime = time.time();
        #do the cosesg
        tmaps = myCoseg(class_ims, class_gt_masks, class_min_fg_areas, class_max_fg_areas, class_min_fg_lengths, class_min_fg_heights, options)
        new_class_cosegs = cellfun(@(x)(x==3 | x==1) , tmaps, 'uniformoutput', false);
        coseg_elapsed = time.time() - stime
        print('Coseg time class {}: {} sec/im.\n'.format(test_class, coseg_elapsed / len(class_ims)))

        for i in range(len(im_inds)):
            ind_str = str(im_inds[i])
            #save_fname = #save that file
            seg = new_class_cosegs[i]
            #save images function
        class_cosegs[class_ind] = new_class_cosegs

    printf('Aggregate and save\n')
    segmentations = np.empty((len(images)))
    for i in range(len(classes))
        test_class = classes[i]
        im_inds = #find([images.class] == test_class)
        segmentations[i] = class_cosegs[i]

    #save(params.coseg_save_fname, 'segmentations', '-v7.3');
    #fprintf('saved to %s\n', save_fname);
