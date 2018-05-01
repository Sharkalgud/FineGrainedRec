

def coseg_main(params):
    coseg_save_dir = params['im_out_dir']

    im_data = open(params['im_meta_fname']).read() #needs to be tweaked based on what is in the file
    images = im_data['images'] #needs modification similar to above line
    classes = list(set(images['classes'])) #broken possibility of using numpy to making operations easier

    
