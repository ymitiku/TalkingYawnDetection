from dataset import DriverActionDataset


def main():
    dataset = DriverActionDataset("/home/mtk/datasets/Yaw/YawDD dataset/Mirror",(227,227,3))
    dataset.video_sequences_to_image_sequences("/home/mtk/datasets/Yaw/YawDD dataset/Mirror","/home/mtk/datasets/Yaw/YawDD dataset/images")

if __name__ == "__main__":
    main()