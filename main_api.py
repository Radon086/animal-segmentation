import warnings
warnings.filterwarnings("ignore")
import sys
import os
import argparse
import cv2
import json
import numpy as np
import pandas as pd
import importlib.util
from ultralytics import YOLO
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
from mmpose.visualization import PoseLocalVisualizer
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
import zipfile
import uuid
import shutil

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

app = FastAPI()
config_path = "./config.py"
config = load_config(config_path)
seg_model = YOLO(config.SEG_MODEL_PATH)
kp_config_file = config.KP_CONFIG_FILE
kp_checkpoint_file = config.KP_CHECKPOINT_FILE
device = config.DEVICE
kp_model = init_model(kp_config_file, kp_checkpoint_file, device= device)  
kp_thr = config.KP_THRESHOLD

def get_distance(a,b):
    a = np.array(a)
    b = np.array(b)
    distance = np.linalg.norm(a - b)
    return distance

def calculate_right_eye_distances(animals, score_threshold=0.3):
    rows = []
    valid_animals = []

    for animal in animals:
        right_eye = animal.get("R_Eye")
        right_score = animal.get("R_Eye_score", 0)

        if right_eye is not None and right_score >= score_threshold:
            valid_animals.append(animal)

    for i in range(len(valid_animals)):
        for j in range(i + 1, len(valid_animals)):
            animal_a = valid_animals[i]
            animal_b = valid_animals[j]

            eye_a = np.asarray(animal_a["R_Eye"], dtype=float)
            eye_b = np.asarray(animal_b["R_Eye"], dtype=float)

            distance = np.linalg.norm(eye_a - eye_b)

            rows.append({
                "animal_A": 'Animal_'+ str(animal_a["animal_id"]),
                "animal_B": 'Animal_'+ str(animal_b["animal_id"]),
                "right_eye_distance": float(distance)
            })

    return pd.DataFrame(rows)

def run_analysis(image_path, output_dir):
    results = seg_model(image_path) 
    result = None
    if results:
        result = results[0]

    bboxes = None
    if result:
        image = cv2.imread(image_path)
        bboxes = result.boxes.xyxy.cpu().numpy().astype(int)
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            ROI = image[y1:y2, x1:x2]
            # debug
            # cv2.imwrite(f"./output/ROI_{i}.jpg", ROI)
            cv2.putText(image, f"Animal {i}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        contours = result.masks.xy
        for contour in contours:
            contour = contour.astype(np.int32)
            cv2.polylines(image,[contour],isClosed=True,color=(0, 255, 0),thickness=2)
        cv2.imwrite(os.path.join(output_dir, "outline.jpg"), image)

    pose_results = None
    if bboxes is not None:
        pose_results = inference_topdown(kp_model, image_path, bboxes=bboxes)
        #print(pose_results)

    if pose_results is None:
        raise ValueError("We can't find kp in the image!")

    data_sample = merge_data_samples(pose_results)
    visualizer = PoseLocalVisualizer()
    visualizer.set_dataset_meta(kp_model.dataset_meta)

    rgb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    visualizer.add_datasample(
        'animal_pose',
        rgb_image,
        data_sample,
        draw_gt=False,
        draw_pred=True,
        draw_bbox=True,
        show_kpt_idx=True,
        kpt_thr=kp_thr,
        out_file= os.path.join(output_dir,"animal_pose.jpg")
    )

    keypoints = data_sample.pred_instances.keypoints
    scores = data_sample.pred_instances.keypoint_scores

    kp_ids = kp_model.dataset_meta['keypoint_name2id']

    animal_meta = []
    for animal_id in range(len(keypoints)):
        meta = {
            'animal_id': animal_id,
            'L_Eye': None,
            "L_Eye_score":None, 
            'R_Eye': None,
            "R_Eye_score": None,
            'Eye_distance': None,
        }

        # get L_eye
        score = scores[animal_id][kp_ids['L_Eye']]
        meta['L_Eye'] = keypoints[animal_id][kp_ids['L_Eye']].tolist()
        meta['L_Eye_score'] = float(score)

        # get R_eye
        score = scores[animal_id][kp_ids['R_Eye']]
        meta['R_Eye'] = keypoints[animal_id][kp_ids['R_Eye']].tolist()
        meta['R_Eye_score'] = float(score)

        # debug
        '''
        for kp_id, (x, y) in enumerate(keypoints[animal_id]):
            score = scores[animal_id][kp_id]
            print(kp_id, x, y, score)
        '''

        if meta['L_Eye_score'] >= kp_thr and meta['R_Eye_score']>=kp_thr:
            meta['Eye_distance'] = get_distance(meta['R_Eye'],meta['L_Eye'])
        animal_meta.append(meta)

    with open(os.path.join(output_dir,"animal_meta.json"), "w", encoding="utf-8") as f:
        json.dump(animal_meta, f, indent=2)

    # print(animal_meta)
    df = calculate_right_eye_distances(animals=animal_meta,score_threshold=kp_thr)
    print(f"Right Eye distance between animals. Threshold = {kp_thr}")
    print(df)
    df.to_csv(os.path.join(output_dir,"right_eye_distances.csv"), index=False)

    return df.to_dict(orient="records")

def create_result_zip(zip_dir):
    zip_path = os.path.join(zip_dir,"results.zip")
    right_eye_distances_path = os.path.join(zip_dir,"right_eye_distances.csv")
    animal_meta_path = os.path.join(zip_dir,"animal_meta.json")
    animal_pose_path = os.path.join(zip_dir,"animal_pose.jpg")
    outline_path = os.path.join(zip_dir, "outline.jpg")

    with zipfile.ZipFile(zip_path, "w") as zip_file:
        zip_file.write(outline_path, "outline.jpg")
        zip_file.write(animal_pose_path,"animal_pose.jpg")
        zip_file.write(animal_meta_path,"animal_meta.json")
        zip_file.write(right_eye_distances_path,"right_eye_distances.csv")
    return zip_path

@app.post("/analyze")
async def analysis_api_call(background_tasks: BackgroundTasks,file: UploadFile = File(...)):

    request_dir = f"./api_output/{uuid.uuid4().hex}"
    os.makedirs(request_dir)
    image_path = os.path.join(request_dir, "tmp.jpg")

    with open(image_path, "wb") as f:
        f.write(await file.read())
    
    result = run_analysis(image_path, request_dir)
    zip_path = create_result_zip(request_dir)
    background_tasks.add_task(shutil.rmtree, request_dir)
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="animal_analysis_results.zip"
    )

