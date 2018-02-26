import os
import shutil

def split_array(array,max_size):
    output = []
    for i in range(0,len(array),max_size):
        output+=[array[i:i+max_size]]
    return output
def copy_images(imgs_files,source_folder,dest_folder):
     for imfile in imgs_files:
         shutil.copy(os.path.join(source_folder,imfile),os.path.join(dest_folder,imfile))

def split_sequence(dataset_dir,output_dir,max_size):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    sequences = os.listdir(dataset_dir)
    for s in sequences:       
        current_path  = os.path.join(dataset_dir,s)
        s_images = os.listdir(current_path)
        s_images.sort()
        splited_seq = split_array(s_images,max_size) 
        for i in range(len(splited_seq)):
            dest_folder = os.path.join(output_dir,s+"-"+str(i))
            if not os.path.exists(dest_folder):
                os.mkdir(dest_folder)
            copy_images(splited_seq[i],current_path,dest_folder)
        print "Processed",s