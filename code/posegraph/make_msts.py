import os
import pickle
def make_msts(config):

    save_fname = config.mst_save_fname
    if os.path.exists(save_fname):
        print('Already did msts')
        return

    out_dir = os.path.abspath(save_fname)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print('Computing msts..')

    #Load features
    print('Loading features ...')
    feat_fname = config.cnn_bbox_fname
    with open(feat_fname, 'r') as f:
        feat_data = pickle.load(f)
        f.close()
    features = #cat(2, feat_data.all_feats{:}); #replace
    features = # elemen wise divide: bsxfun(@rdivide, features, sqrt(sum(features.^2, 1))); % L2-normalize

    print('Compute distances')
    dists = mypdist()

    num_msts = config.num_msts;

    dists =  mypdist(features.T) #broken

    num_msts = config.num_msts

    dists = sparse(double((dists + dists')/2));
    msts = []
    for i in range(num_msts):
        print('mst {}/{}'.format(i, num_msts))
        #tree = mst(dists)
        mists.append(tree)
        dists(find(tree)) = 0 #def broken

    with open(save_fname, 'wb') as f:
        pickle.load(msts, f)
        f.close()
