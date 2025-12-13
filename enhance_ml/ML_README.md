# ECE_253_Project - ML-Models

A. Zero-DCE

    How to implement:
    1. clone the repo
    2. create conda env by: conda create --name zerodce_env opencv python=3.7 -c pytorch
    3. then install pytorch and torchvision manually: pip install torch==1.0.0 torchvision==0.2.1
    4. run the test code: python Zero-DCE_code/lowlight_test.py
    5. test results will be generated in Zero-DCE_code/data/result 


B. NAFNet (run on Colab)
    To run the code:
    option (a) check the link: https://colab.research.google.com/drive/15fqzFiVsbzOf0899CRodVAPBctVW4Wd_?usp=sharing
    option (b) store on your Google Drive and run Google Colab


C. FBCNN (ML_restore.py)
    ML_restore.py is for restoring the compressed images by using the FBCNN, which can be found on https://github.com/jiaxi-jiang/FBCNN.git. 
    The .pt file can be found on https://github.com/jiaxi-jiang/FBCNN/releases/tag/v1.0