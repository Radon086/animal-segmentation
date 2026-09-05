# 1. Project Introduction

This project detects multiple animals in an image and shows their segmentation outlines. After assigning each detected animal an ID, the
system calculates the left-right eye distance of each animal and the
right-eye distance between every pair of animals.

Notes:

1. A measurement is calculated only when the required keypoint
   confidence scores pass the configured threshold.
2. The current measurement is performed in pixel space. The distance is
   calculated using L2 (Euclidean) pixel distance. No physical-unit
   measurement such as mm is claimed.
3. Metadata, segmentation-outline visualization, and keypoint
   visualization are provided for result verification.
4. To meet the assignment requirements, the same analysis pipeline is
   provided in three execution forms: local execution, FastAPI API
   execution, and Docker deployment.

# 2. Measurement Method

The eye positions are represented as 2D pixel coordinates:

    P = (x, y)

The distance between two points is calculated using the L2 (Euclidean)
distance:

    d(P1, P2) = sqrt((x1 - x2)^2 + (y1 - y2)^2)

For each animal:

    Eye_distance = d(L_Eye, R_Eye)

For two different animals:

    Right_eye_distance = d(R_Eye_A, R_Eye_B)

A measurement is only generated when the required keypoint confidence
scores are above the configured threshold.

The current implementation measures distance in pixel space. Converting
the result to a physical unit such as millimeters would require camera
calibration or another known scale reference.

# 3. Main steps

```
Flow Chart
                 Input Image
                      │
                      ▼
              FastAPI /analyze
                      │
                      ▼
  YOLO26s-seg Animal Detection + Segmentation
                      │
             Animal Bounding Boxes
                      │
                      ▼
          MMPose Animal-Pose HRNet
                 Eye Keypoints
                      │
             ┌────────┴────────┐
             ▼                 ▼
      L/R eye distance    R-eye ↔ R-eye
        per animal          between animals
             │                 │
             └────────┬────────┘
                      ▼
                JSON / CSV
                      │
                      ▼
               ZIP Response
```
```
When deployed with Docker:
┌───────────────────────────────────┐
│ FastAPI + Uvicorn                 │
│   └── Analysis Pipeline           │
│       ├── YOLO26s-seg             │
│       ├── MMPose HRNet-W32         │
│       └── Measurement              │
└───────────────────────────────────┘
              ▲
              │ HTTP
              ▼
          Client / Browser
```
1. Use a segmentation model to identify animals and their outlines. 
I picked Ultralytics YOLO26s-seg because of its relatively small model
size and good inference speed, while combining object detection and segmentation in one model.
2. Use an Animal Pose Model to estimate each animal's left and right eye
positions. I picked MMPose because it provides a dedicated animal pose estimation model.
3. Calculate the distance between the left and right eyes of each animal.
4. Calculate the distance between the right eyes of every pair of animals.
5. Export measurement results, metadata, CSV data, and visualization images.
6. FastAPI provides the HTTP API, while Docker packages the complete runtime environment for deployment.

# 4. Tech stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.8 |
| Segmentation | Ultralytics YOLO26s-seg |
| Animal Pose Estimation | MMPose Animal-Pose HRNet-W32 |
| Detection Framework | MMDetection 3.3.0 |
| Deep Learning Framework | PyTorch 2.4.1 |
| CUDA | CUDA 12.1 |
| API Framework | FastAPI |
| API Server | Uvicorn |
| Image Processing | OpenCV |
| Data Processing | NumPy / Pandas |
| Container | Docker |
| GPU Runtime | NVIDIA Container Toolkit |

# 5. Environment
Create the Conda environment:
```bash
conda create -n animal_seg python=3.8 pip -y
conda activate animal_seg
```
Install PyTorch:
```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```
Install MMCV:
```bash
pip install mmcv==2.1.0
```
Install the remaining dependencies:
```bash
pip install -r requirements.txt
```

# 6. How to run locally
Activate the Conda environment:
```bash
conda activate animal_seg
```
Run the analysis:
```bash
python main.py --img <your_img_path>
```
The generated results will be saved in:
```bash
./output folder
```
# 7. How to Run the API
Start the API server:
```bash
conda activate animal_seg
uvicorn main_api:app --reload
```
Open the FastAPI Swagger UI in your browser:
```bash
http://127.0.0.1:8000/docs
```
Then:
```
Open POST /analyze.
Click Try it out.
Upload an image.
Click Execute.
```
The API returns a ZIP file containing the analysis results.
# 8. How to Run with Docker
Build the Docker image:
```bash
docker build -t animal-segmentation .
```
Run the container:
```bash
docker run --rm --gpus all -p 8000:8000 animal-segmentation
```
Open the FastAPI Swagger UI:
```bash
http://localhost:8000/docs
```
Then:
```
Open POST /analyze.
Click Try it out.
Upload an image.
Click Execute.
```
The API returns a ZIP file containing the analysis results.

# 9. Testing account info
Not needed for this project.
The API does not require user authentication, registration, or third-party API keys.