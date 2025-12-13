# ECE_253_Project - ML-Models

## A. Zero-DCE
How to implement:
1. clone the repo
2. create conda env by: 
`conda create --name zerodce_env opencv python=3.7 -c pytorch`
3. then install pytorch and torchvision manually: `pip install torch==1.0.0 torchvision==0.2.1`
4. `cd Zero-DCE_code` & run the test code: `python lowlight_test.py`
5. test results will be generated in `Zero-DCE_code/data/result` 


## B. NAFNet (run on Colab)
To run the code:
1. - option (a) check the link: [online colab](https://colab.research.google.com/drive/15fqzFiVsbzOf0899CRodVAPBctVW4Wd_?usp=sharing)
    - option (b) store on your Google Drive and run Google Colab
2. Run code following instructions of comments



## C. FBCNN (ML_restore.py)
- `ML_restore.py` is for restoring the compressed images by using the FBCNN, which can be found on [FBCNN Github](https://github.com/jiaxi-jiang/FBCNN.git). 
- The .pt file can be found on [FBCNN v1.0](https://github.com/jiaxi-jiang/FBCNN/releases/tag/v1.0)