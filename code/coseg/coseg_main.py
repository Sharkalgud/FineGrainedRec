import numpy as np
import scipy as sp
import time as time
import pickle
import cv2
import grab_cut

#might have to define python object to hold meta date for images
def coseg(params):
    coseg_save_dir = params['im_out_dir']

    with open(params['im_meta_fname']) as f:
        images = pickle.load(f)
        f.close()
    classes = list(set([i['class'] for i in images]))

    #options should be modified to work with new wrapper
    options = {}
    options['num_iters'] = params['coseg_iters']
    options['class_weight'] = params['class_weight']
    options['use_class_fg'] = params['use_class_fg']
    options['use_class_bg'] = params['use_class_bg']
    options['fg_prior'] = params['fg_prior']
    options['do_refine'] = params['do_refine']

    class_cosegs =  []

    for class_ind in range(len(classes)):
        test_class = classes[class_ind]
        print('Coseg class {}\n'.format(test_class))

        #Load images and set them up for coseg_main
        stime = time.time()
        im_inds = [i for i in range(len(images)) if images[i]['class'] == test_class]
        class_ims = []
        class_gt_masks = []
        # class_min_fg_areas = [0 for i in range(len(im_inds))]
        # class_max_fg_areas = [1000000 for i in range(len(im_inds))]
        # class_min_fg_lengths = [0 for i in range(len(im_inds))]
        # class_min_fg_heights = [0 for i in range(len(im_inds))]
        recs = [0 for i in range(len(im_inds))]
        for i in range(len(im_inds)):
            ind_str = str(im_inds[i])

            #get image
            im = np.imread(os.path.join(params['im_base'], images[im_inds[i]]['rel_path']))
            if np.size(im, 2) == 1:
                im = np.stack([im, im, im], axis = 2)
            scale = (params['resize_area'] / (np.size(im, axis = 0) * np.size(im, axis = 1))) ** (0.5)
            im_scaled = sp.misc.resize(im, scale)
            class_ims.append(im_scaled)

            #set up GT mask using the bounding box
            class_gt_masks.append(10 * np.ones((np.size(im_scaled, axis = 0), np.size(im_scaled, axis = 1)), dtype=int))
            #Bbox init
            bbox = images[im_inds[i]]['bbox'];
            x1 = 0#max(0, int((bbox.x1-1)*scale+1 - params['bbox_context']));
            x2 = 0#min(np.size(im_scaled, 1), int((bbox.x2-1)*scale+1 + params['bbox_context']));
            y1 = np.size(im_scaled, axis = 0)#max(0, int((bbox.y1-1)*scale+1 - params['bbox_context']));
            y2 = np.size(im_scaled, axis = 1)#min(np.size(im_scaled, 0), round((bbox.y2-1)*scale+1 + params['bbox_context']));
            recs[i] = (x1, x2, y1, y2)
            #GC_BGD = 0, GC_FGD = 1, GC_PR_BGD = 2, GC_PR_FGD = 3
            # class_gt_masks[i][:, :] = 0;
            # class_gt_masks[i][y1:y2, x1:x2] = 3
            # if not (0 in class_gt_masks[i]):
            #     class_gt_masks[i][0, :] = 0;
            #     class_gt_masks[i][:, 0] = 0;
            #     class_gt_masks[i][-1, :] = 0;
            #     class_gt_masks[i][:, -1] = 0;
            # assert 3 in class_gt_masks[i]
            # assert 0 in class_gt_masks[i]
            # bbox_length = x2 - x1 + 1
            # bbox_height = y2 - y1 + 1
            # bbox_area = bbox_length * bbox_height
            # class_min_fg_areas[i] = params['bbox_min_fg_area'] * bbox_area;
            # class_max_fg_areas[i] = params['bbox_max_fg_area'] * bbox_area;
            # class_min_fg_lengths[i] = params['bbox_min_fg_length'] * bbox_length;
            # class_min_fg_heights[i] = params['bbox_min_fg_height'] * bbox_height;
        elapsed = time.time() - stime
        print('Loading time class {}: {} sec\n'.format(test_class, elapsed))

        stime = time.time();
        #do the cosesg
        tmaps = myGrabCut(class_ims, recs, params['coseg_iters'])
        #new_class_cosegs = cellfun(@(x)(x==3 | x==1) , tmaps, 'uniformoutput', false);
        coseg_elapsed = time.time() - stime
        print('Coseg time class {}: {} sec/im.\n'.format(test_class, coseg_elapsed / len(class_ims)))

        for i in range(len(im_inds)):
            ind_str = str(im_inds[i])
            save_fname = os.path.join(coseg_save_dir, ind_str)
            seg = new_class_cosegs[i]
            with open(save_fname, 'wb') as f:
                pickle.dump(seg, f)
                f.close
        class_cosegs.append(new_class_cosegs)

    printf('Aggregate and save\n')
    segmentations = [0 for i in range(len(images))]
    for i in range(len(classes)):
        test_class = classes[i]
        im_inds = [i for i in range(len(images)) if images[i]['class'] == test_class]
        segmentations[im_inds] = class_cosegs[i]

    #save(params.coseg_save_fname, 'segmentations', '-v7.3');
    with open(params['coseg_save_fname'], 'wb') as f:
        pickle.dump(segmentations, f)
        f.close()

    #fprintf('saved to %s\n', save_fname);
    print('saved to file')
