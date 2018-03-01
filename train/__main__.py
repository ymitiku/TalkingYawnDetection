from dataset.mouth_features import MouthFeatureOnlyDataset
from nets.mouth_features import MouthFeatureOnlyNet


def main():
    dataset = MouthFeatureOnlyDataset("/dataset/yawn/splitted-100","/dataset/yawn/tracked",(24,24,3),100)
    dataset.load_dataset()
    net = MouthFeatureOnlyNet(dataset,(24,24,3),100)
    net.train()
if __name__ == "__main__":
    main()