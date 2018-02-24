from dataset import DriverActionDataset
from nets import Network


def main():
    dataset = DriverActionDataset("/dataset/yawn/splited-100",(24,24,3),100)
    dataset.load_dataset()
    net = Network(dataset,(24,24,3),100)
    net.train()
if __name__ == "__main__":
    main()