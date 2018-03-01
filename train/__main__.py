from dataset import DriverActionDataset
from nets import Network


def main():
    dataset = DriverActionDataset("/dataset/yawn/splitted-30",(24,24,3),30)
    dataset.load_dataset()
    net = Network(dataset,(24,24,3),30)
    net.train()
if __name__ == "__main__":
    main()