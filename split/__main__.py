from split import get_cmd_args
from split.split_squence import track_faces_inside_sequences,split_sequence
def main():
    args = get_cmd_args()
    print "tracking all faces"
    track_faces_inside_sequences(args.images_path,args.faces_path)
    print "done with tracking faces"
    print "splitting dataset"
    split_sequence(args.images_path,args.output_path,args.sequence_length)
if __name__ == '__main__':
    main()