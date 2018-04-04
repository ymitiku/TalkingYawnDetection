from dataset.mouth_features import MouthFeatureOnlyDataset
from nets.mouth_features import MouthFeatureOnlyNet
import argparse
def get_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset_path",type=str)
    parser.add_argument("-f","--faces_path",type=str)
    parser.add_argument("-l","--sequence_length",type=int,default=30)
    args = parser.parse_args()
    return args
def main():
    args  = get_cmd_args()
    dataset = MouthFeatureOnlyDataset(args.dataset_path,args.faces_path,(48,48,1),30)
    dataset.load_dataset()
    net = MouthFeatureOnlyNet(dataset,(48,48,1),30)
    net.train()
if __name__ == "__main__":
    main()