from dataset import DriverActionDataset
import dlib

def main():
    dataset = DriverActionDataset("/home/mtk/datasets/Yaw/YawDD dataset/Mirror",(227,227,3))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    dataset.load_image_sequence("/dataset/yawn/images3/9-FemaleNoGlasses-Normal/",detector,predictor)

if __name__ == "__main__":
    main()