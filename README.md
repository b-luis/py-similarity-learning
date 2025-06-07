# py-similarity-learning
Facial recognition using a similarity learning method with ArcFace and YOLO-based face detection.

## 🧬 How It Works: Similarity Learning & Facial Embeddings

This project uses a **Similarity Learning** approach to perform facial recognition, instead of traditional classification.

### 🧠 What is Similarity Learning?

In traditional classification, a model learns to predict a fixed set of classes (e.g. "Alice", "Bob", "Charlie"). However, this doesn't scale well for real-world use where new people can be added anytime.

**Similarity Learning** solves this by learning how to **compare faces** rather than classify them.

- The model learns to say:  
  "**How similar is this face to another face?**"  

  Instead of:  
  "**Which person is this?**"

1. **Face Detection**  
   YOLOv5 (trained on faces) detects and crops the face from the image.

2. **Facial Embedding Extraction**  
   A deep neural network (iResNet100 with ArcFace loss) converts each face into a **128D or 512D vector**, called a **facial embedding**.

3. **Embedding Comparison**  
   When a new face is captured:
   - It is converted into an embedding.
   - That embedding is compared to existing embeddings using a similarity metric (like **cosine similarity** or **Euclidean distance**).
   - If it's similar enough to a known face, it’s recognized.

## 👾 Usage
Clone repository
```
git clone https://github.com/b-luis/py-similarity-learning.git
cd py-similarity-learning
```  

🛠️ Requirements
```
pip install -r requirements.txt
```

📦 Download weights
| Pre-trained               | Backbone            | **Description**                                                                | Download link                                                                                | Size     |
|:--------------------------|:--------------------|:-------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------|:---------|
| backbone.pth              | iresnet100          | for feature extraction with arcface loss function. save in backbones folder    | [File](https://drive.google.com/file/d/1TVfnDTCYa1bS9Yat-h2SAos0qjAwN3vI/view?usp=drive_link)| 249.1 MB |
| yolov5m-face.pt           | YOLO5-CSPNet        | used for face detection. save in weights folder                                | [File](https://drive.google.com/file/d/1bP86MtZNFQ-c8dgYf_-UuGAthnzSdmFr/view?usp=drive_link)| 161.3 MB |
| yolov5s-face.pt           | YOLO5-CSPNet        | used for face detection, but much smaller in size. save in weights folder      | [File](https://drive.google.com/file/d/11oKjCKTVVTXqX5T9GJ9mdPAxCu9eZS2S/view?usp=drive_link)| 54.4 MB  |
 

📁 Overall directory structure

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

## 🧠 Training

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
## ⚡ GPU Acceleration (CUDA)

To speed up training and recognition, a CUDA-compatible NVIDIA GPU is recommended.

- **Windows/Linux (NVIDIA GPU)**:
  - Install the correct [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and cuDNN.
  - Verify by running `torch.cuda.is_available()` in Python.
  - This significantly improves performance over CPU.

- **macOS**:
  - CUDA is **not supported**.
  - Training and recognition will run on **CPU** only, which is slower.
  - Works fine for small datasets or light usage.

- **AMD GPUs / Apple Silicon**:
  - Currently not supported for GPU acceleration in this project.

 - Verify `torch.cuda.is_available()` returns True in Python to confirm CUDA is working.
     
> 💡 Without CUDA, the training and recognition processes will fall back to CPU and can be significantly slower.
