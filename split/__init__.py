import argparse

def get_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--images_path",type=str)
    parser.add_argument("-f","--faces_path",type=str)
    parser.add_argument("-o","--output_path",type=str)
    parser.add_argument("-l","--sequence_length",type=int,default=30)
    args = parser.parse_args()
    return args
