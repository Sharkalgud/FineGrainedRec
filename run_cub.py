from set_config import set_config
from run_coseg import run_coseg

domain = 'cub';
config = set_config(domain)

make_cub_anno_files(config) #modify annotation files to work with program
run_coseg(config) #run cosegmentation methodology on dataset

#can add semgentation evaluation stuff if needed

extract_feats_bbox_domain(config)
