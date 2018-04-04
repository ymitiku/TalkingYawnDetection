# Talking & Yawn Detection
This project is aimed to train model that detects talking and yawning from sequential images.

## How to split dataset from videos
This proejct uses[Yawdd dataset](http://www.site.uottawa.ca/~shervin/yawning/). 
* First modify dataset/extract.sh file to give mirror folders for both male and female subjects that contain extracted videos. Also modify the output directory(dataset/yawn/images3/)
* Second split he extraced sequence images to smaller sequences by running the following command
``` python -m split --images_path path-to-extracted-images --faces_path path-to-save-bounding-boxes-of-sequence-images --output_path path-to-save-output-sequences --sequence_length sequence-length-to-split```

### How to run training program

``` python -m train --dataset_path path-to-splitted-dataset --faces_path pathes-to-bounding-boxes --sequence_length sequence-length ```

* **shape_predictor should be inside root directory of this project. Shape predictor can be downloaded to project using the following script.**
```
cd /path-to-project
wget "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

 [sp]: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2