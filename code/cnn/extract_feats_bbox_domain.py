import tempfile
import os
import pickle
import time

def extract_feats_bbox_domain(config):
    #User for the pose the pose pose_graph
    #For train, test
    #Crop each image to bbox + padding
    #Extract the specified features. Flips/Crops?
    #Make extra labels if we need to
    #Save to a file

    options = {}

    options['pix_padding'] = config.cnn_padding
    options['layer'] = config.pose_graph_layer

    options['use_whole'] = True
    options['use_center'] = False
    options['use_corners'] = False
    options['use_flips'] = False

    options['use_gpu'] = True
    options['gpu_num'] = config.gpu_num
    options['temp_model_def_loc'] = tempfile.NamedTemporaryFile().name

    options['net'] = 'caffenet'  #this needs to be changed to pytorch
    options['mean_fname'] = config.ilsvrc_mean_loc
    options['model_def_file'] = config.caffenet_deploy_loc
    options['model_file'] = config.caffenet_model_loc

    split = 'train'

    save_fname = config.cnn_bbox_fname

    if os.path.exists(save_fname, 'file'):
        print('Already done!')
        return

    #load
    print('Load images...')
    with open(config.train_imagedata_fname, 'r') as f:
         images = pickle.load(f)
         ims = pickle.load(f)
         f.close()

    print('done')
    labels = [im['class'] for im in images]
    all_feats = [] #maybe need to be changed
    all_labels = [] #maybe need to be changed
    s = time.time()
    for i in range(len(images)):
        if i % 256 == 1:
            print('cnn feats {} {}/{}'.format(split, i, len(images)))
            s = time.time()

        #Crops
        im = ims[i]
        bbox = images[i]['bbox']
        x1 = max(0, bbox.x1 - options['pix_padding'])
        x2 = min(np.size(im, axis = 1), bbox.x2, + options['pix_padding']) #might be an off by one type thing
        y1 = max(0, bbox.y1 - options['pix_padding'])
        y2 = min(np.size(im, axis = 0), bbox.y2, + options['pix_padding'])
        im = im[y1 : y2, x1 : x2, :]

        #Extract features
        features = #coming to a stack soon extract_pytorch(im, options)
        if 0 in np.size(features): #dependent on  how I extract features
            print('bad layer')
            continue
        all_feats.append(features) #also iffy
        all_labels = [images[i]['class'] for j in np.size(features, axis = 1)]

    pardir = os.path.abspath(save_fname)
    if not os.path.exists(pardir):
        os.makedirs(pardir)
    with open(save_fname, 'wb') as f:
        pickle.dump(all_feats)
        pickle.dump(all_labels)
        pickle.dump(options)

    if os.path.exists(options['temp_model_def_loc'])
        os.remove(options['temp_model_def_loc'])
