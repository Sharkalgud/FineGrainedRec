#file to set paths to important parts of the implementation
import os
class Config(): __slots__ = '''cub_root cub_seg_dir car_root
                            caffe_root pytorch_root rcnn_root
                            liblinear_dir gpu_num train_image_fname root
                            train_image_fname pose_graph_layer'''

def set_config(domain):
    ########
    # Input: domain: name of dataset being used
    # Output: dictionary with config paths
    ########
    config = Config()

    # Location of your CUB_200_2011 directory
    config.cub_root = ''

    #cosegmentation directory for CUB. Necessary if evaluating cosegmentation
    config.cub_seg_dir = ''

    # Location of the directory containing cars_annos.mat probably won't need this
    config.car_root = ''

    # Location of caffe install. Won't be needed when pytorch is implemented
    config.caffe_root = ''

    # Location of pytorch install.
    config.pytorch_root = ''

    #Location of rcnn install. Don't know if needed right now
    config.rcnn_root = ''

    #Location of liblinear-dense-float. Don't know if I need this?
    config.liblinear_dir = ''

    #Which gpu to use (0-indexed) this will be implementation specific
    config.gpu_num = 2;

    #### Look at bottom of config file for rest of file ###
    config.root = os.getcwd()
    config.train_image_fname = os.path.join(config.root, 'prcoessed', 'data', 'cub_images_train.mat');
    config.pose_graph_layer = 'conv4'
    config.cnn_bbox_fname = os.path.join(config.root, 'processed', 'cnn', 'features', '{}_{}_train.mat'.format(config.pose_graph_layer, domain));

    return config
