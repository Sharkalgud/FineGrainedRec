from FineGrainedRec.set_config import set_config
from FineGrainedRec.code.coseg.run_coseg import run_coseg
from FineGrainedRec.code.scripts.make_cub_anno_files import make_cub_anno_files

domain = 'cub';
config = set_config(domain)

make_cub_anno_files(config) #modify annotation files to work with program
run_coseg(config) #run cosegmentation methodology on dataset

#can add semgentation evaluation stuff if needed

#extract_feats_bbox_domain(config)
#make_msts(config)

#run_matching(config)
#part_propagate(config)
#tighten_parts(config)

#my_run_rcnn(config)

#vgg_fibetune(config)

#run_recognition(config)
