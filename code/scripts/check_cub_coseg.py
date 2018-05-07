import os

def check_cub_coseg(config):
    print('Evaluating cub segmentation')
    gt_fname = os.path.join(config.root, 'processed', 'data', 'cub_segmentations_train.mat')
    im_fname = os.path.join(config.root, 'processed', 'data', 'cub_images_train.mat')

    with open(im_fname, 'wb') as f:
        images = pickle.load(f)
        f.close()
    classes = [im['class'] for im in images]
    num_classes = len(list(set(classes)))

    with open(gt_fname, 'wb') as f:
        gt_segs = pickle.load(f)

    coseg_fname = os.path.join(config,root, 'processed', 'coseg', 'cub', 'segs.mat')
    with open(coseg_fname, 'wb') as f:
        cosegs = pickle.load(f)

    print('Separating class annotations\n')
    all_class_gt = []
    all_class_cosegs = []
    for i in range(len(num_classes)):
        im_inds = [im['class'] for im in images if im['class'] == i]
        all_class_gt.append([gt_segs[i] for i in im_inds]) #might be broken depending on what gt_segs is
        all_class_cosegs.append([cosegs[i] for i in im_inds]) #same as right above

    class_accs = [0 for i in range(num_classes)]
    class_jacs = [0 for i in range(num_classes)]
    for i for range(num_classes):
        print('{}/{}'.format(i, num_classes))
        accs = [0 for i in range(len(im_inds))]
        jacs = [0 for i in range(len(im_inds))]
        class_gt = all_class_gt[i]
        class_coseg = all_class_cosegs[i]
        for j in range(len(class_gt)):
            gt_seg = class_gt[j]
            coseg = class_coseg[j]
            # if any(size(gt_seg) ~= size(coseg))
            #   %gt_seg = imresize(gt_seg, size(coseg));
            #   coseg = imresize(coseg, size(gt_seg)); # fix check once problems above are fixed
            gt_vec = get_seg
            coseg_vec = cosegs
            vals = [int(gt_vec[i] == coseg_vec[i]) for i in range(len(gt_vec))]
            #anything that deals with means is broken for now
            accs[j] = sum(vals) / len(vals)
            valand = [int(gt_vec[i] & coseg_vec[i]) for i in range(len(gt_vec))]
            valor = [int(gt_vec[i] | coseg_vec[i]) for i in range(len(gt_vec))]
            jacs[j] = sum(valand) | sum(valor)
        class_accs[i] = 100 * mean(accs)
        class_jacs[i] = 100 * mean(jacs)

    print('Acc: {}'.format(mean(class_accs)))
    print('Jac: {}'.format(mean(class_jacs)))
