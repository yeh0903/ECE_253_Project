Dataset D can be found on https://universe.roboflow.com/cell-phone-v3g3u/cell-phone-z59qs/dataset/1

Access to dataset D' is on https://drive.google.com/drive/folders/14v7EyqaY3exsCvuAU_TFCJRZCdkb4RLV?usp=drive_link

Please modify the code according to the local path where you download the data.

graph_result.ipynb is for graphing all the test results and the loss curve of the fine tuning models

compress.py makes the raw images we collected compressed

ML_restore.py is for restoring the compressed images by using the FBCNN, which can be found on https://github.com/jiaxi-jiang/FBCNN.git. The .pt file can be found on https://github.com/jiaxi-jiang/FBCNN/releases/tag/v1.0
