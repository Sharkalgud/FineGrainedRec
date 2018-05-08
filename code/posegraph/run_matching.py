import os
import pickle

def run_matching(config):

    bbox_context = 0.5
    resize_height = 150
    try_flip = True
    mst_fname = config.mst_save_fname
    seg_fname = config.coseg_save_fname

    im_out_dir = config.matching_im_out_dir;
    imdata_fname = config.train_imagedata_fname;
    im_base = config.im_base;

    save_fname = config.alignment_fname;

    if os.path.exists(save_fname):
        print('Matching alread done!')
        return

    if not os.path.exists(im_out_dir):
        os.mkdirs(im_out_dir)

    #get the pose pose_graph_layer
    print('Matching, loading files...')
    with open(mst_fname, 'r') as f:
        msts = pickle.load(f)
        graph = msts[i]
        for i in range(1, len(msts)):
            graph += msts[i]

    #Load segmenetations and images
    with open(seg_fname, 'r') as f:
        segmentations = pickle.load(f)
        f.close()
    with open(imdata_fname, 'r') as f:
        images = pickle.load(f)
        all_ims = pickle.load(f)
        f.close()
    print('Done loading')

    for i in range(len(images)):
        print('Matching {}{}'.format(i, len(images)))
        out_fname = os.path.join(im_out_dir, str(i) + '.p')
        if os.path.exists(out_fname):
            continue

        start = i
        to_inds = find(graph(from,:)>0) #def broken

        inds = [start] + to_inds
        local_images = [images[i] for i in inds]
