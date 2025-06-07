# py-similarity-learning
## Description
Facial Recognition using Similarity Learning method

## Usage
Clone repository
```
git clone https://github.com/b-luis/py-similarity-learning.git
cd py-similarity-learning

```  

Requirements
```
pip install -r requirements.txt
```

Download weights
| Pre-trained               | Backbone            | **Description**                                                                | Download link                                                                                | Size     |
|:--------------------------|:--------------------|:-------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------|:---------|
| backbone.pth              | iresnet100          | for feature extraction with arcface loss function. save in backbones folder    | [File](https://drive.google.com/file/d/1TVfnDTCYa1bS9Yat-h2SAos0qjAwN3vI/view?usp=drive_link)| 249.1 MB |
| yolov5m-face.pt           | YOLO5-CSPNet        | used for face detection. save in weights folder                                | [File](https://drive.google.com/file/d/1bP86MtZNFQ-c8dgYf_-UuGAthnzSdmFr/view?usp=drive_link)| 161.3 MB |
| yolov5s-face.pt           | YOLO5-CSPNet        | used for face detection, but much smaller in size. save in weights folder      | [File](https://drive.google.com/file/d/11oKjCKTVVTXqX5T9GJ9mdPAxCu9eZS2S/view?usp=drive_link)| 54.4 MB  |
 

Overall directory structure

```
├── dataset                      # for training
│   ├── add-train-data           # add photos
│   ├── face-datasets            # all datasets are saved here
│   └── trained-data             # all trained datasets are saved here
├── model_arch                   # where models are saved
│   ├── backbones # Model Weights
│   │   ├── __init__.py
│   │   ├── backbone.pth         # save the weight here
│   │   └── iresnet.py           # resnet architecture
├── static
│   ├── features.npz             # facial embeddings. will be used for similarity matching.
├── yolov5_face                  # cloned from yolov5face repository (unnecessary files are removed)
│   ├── models
│   │   ├── __init__.py
│   │   ├── common.py
│   │   ├── yolo.py
│   │   ├── experimental.py
│   │   └── other yaml files..
│   ├── utils
│   │   ├── __init__.py
│   │   ├── activations.py
│   │   ├── autoanchor.py
│   │   ├── datasets.py
│   │   └── other util files..
│   └── detect_face.py
├── fr.py                        # facial recognition
├── train.py                     # train added dataset
```

## Training

Create folders inside the add-train-data folder
```
├── dataset                      
│   ├── add-train-data           
│   │   ├── lastname_firstname
│   │   │   ├── pic1.jpg
│   │   │   ├── pic2.jpg
│   │   │   └── pic3.jpg
│   │   └── lastname_firstname
│   │       ├── pic1.jpg
│   │       ├── pic2.jpg
│   │       └── pic3.jpg
│   ├── face-datasets
│   └── trained-data   
```
Run in terminal
```
python train.py --is-add-user=True
```
Note: After training an .npz file will be saved inside the static folder.  

Do face recognition
```
python fr.py
```
## ⚡ CUDA Support (Optional but Recommended)

To speed up training and face recognition, it's highly recommended to use a GPU with CUDA support.

Windows Users:
 - Ensure you have a CUDA-compatible NVIDIA GPU.
 - Install the correct versions of CUDA Toolkit and cuDNN.
 - Verify torch.cuda.is_available() returns True in Python to confirm CUDA is working.
     
> 💡 Without CUDA, the training and recognition processes will fall back to CPU and can be significantly slower.
