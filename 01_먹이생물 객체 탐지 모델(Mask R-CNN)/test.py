################################ Args
import argparse
import ast


parser = argparse.ArgumentParser()
parser.add_argument('--classes_names_dict', type=str, help='classes_names_dict')
parser.add_argument('--dataset', type=str, help='dataset path')
parser.add_argument('--bbox_polygon', type=str, help='bbox or polygon')
parser.add_argument('--weights', type=str, help='model path')
parser.add_argument('--save_dir', type=str, help='test result save dir')
parser.add_argument('--dataset_rule', type=str, help='dataset_rule over TVT', default='**/*.json')
parser.add_argument('--label_folder', type=str, help='label_folder', default='2.라벨링데이터')
parser.add_argument('--image_folder', type=str, help='image_folder', default='1.원천데이터')
parser.add_argument('--shuffle', type=int, help='shuffle dataset', default=0)
parser.add_argument('--confidence', type=float, help='confidence', default='0.1')
parser.add_argument('--individual', type=int, help='confidence', default=1)
parser.add_argument('--start_index', type=int, help='test dataset start index', default=1)
parser.add_argument('--last_index', type=int, help='test dataset last index', default=None)
parser.add_argument('--detail_number_print', type=int, help='detail number print', default=1)
parser.add_argument('--IoU', type=float, help='IoU', default=0.5)

ARGS = parser.parse_args()
print("IoU:", ARGS.IoU)


############################### Path

import os
import json
import numpy as np
import skimage
import matplotlib.pyplot as plt

import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn import visualize
import mrcnn.utils as utils

import datetime
import pytz
import glob
from tqdm import tqdm
import random
import time
from tabulate import tabulate
import sys

# 한국 시간대 설정
korea_timezone = pytz.timezone('Asia/Seoul')
date_format = "%Y-%m-%d %H:%M:%S %Z%z"
# 코드 시작 시간 기록
start_time = time.time()

################################ Logger
# 시작 시간을 한국 시간으로 변환
start_korea_time = datetime.datetime.now(korea_timezone).strftime(date_format)
class Logger_all_print_out_txt:
 
    def __init__(self, log_path):
        self.log_path = log_path
        self.console = sys.stdout

        
        
        with open(self.log_path, 'w', encoding='utf-8 sig') as f:
            f.write(f"{start_korea_time}\n\n")
            f.write(f'''!source /content/venv/bin/activate; python "test.py"\\''')
            f.write(f'''\n--classes_names_dict "{ARGS.classes_names_dict}"\\''')
            f.write(f'''\n--dataset "{ARGS.dataset}"\\''')
            f.write(f'''\n--bbox_polygon "{ARGS.bbox_polygon}"\\''')
            f.write(f'''\n--weights "{ARGS.weights}"\\''')
            f.write(f'''\n--save_dir "{ARGS.save_dir}"\\''')
            f.write(f'''\n--IoU {ARGS.IoU}\\''')
            f.write(f'''\n--confidence {ARGS.confidence}\\''')
            f.write(f'''\n--individual {ARGS.individual}\n\n''')
            
            pass
 
    def write(self, message):
        self.console.write(message)
        self.write_onlytxt(message)
        
    def write_onlytxt(self, message):
        with open(self.log_path, 'a', encoding='utf-8 sig') as f:
            f.write(message)
    def flush(self):
        self.console.flush()


class filename_out_txt:
    def __init__(self, log_path):
        self.log_path = log_path
        with open(self.log_path, 'w', encoding='utf-8 sig') as f:
            f.write(f"{start_korea_time}\n\n")

    def write_image_filename(self, image_filename):
        with open(self.log_path, 'a', encoding='utf-8 sig') as f:
            f.write(image_filename + '\n')

    def write_endtime(self, end_korea_time):
        with open(self.log_path, 'a', encoding='utf-8 sig') as f:
            f.write(f'\n{end_korea_time}')

    def flush(self):
        pass

################################ Config
class InferenceConfig(Config):
    

    names_dict_SK = ast.literal_eval(ARGS.classes_names_dict)
    
    NAME = ""
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 1 + len(names_dict_SK)  # BG + classes
    
    USE_MINI_MASK = False

    dataset_link_SK = ARGS.dataset
    
    bbox_polygon_SK = ARGS.bbox_polygon
    
    model_path_SK = ARGS.weights

    result_save_path_SK = ARGS.save_dir + '/images'
    if not os.path.exists(result_save_path_SK):
        os.makedirs(result_save_path_SK)
    else:
        raise Exception(f"{result_save_path_SK} is already exists")

    log_path_SK = ARGS.save_dir + '/log.txt'

    #테스트에 사용된 이미지 파일명 저장(1202 소라 추가)
    imagelist_SK = ARGS.save_dir + '/imagelist.txt'

    DETECTION_MIN_CONFIDENCE = ARGS.confidence

    start_index = ARGS.start_index
    last_index = ARGS.last_index

inference_config = InferenceConfig()

assert inference_config.bbox_polygon_SK in ['bbox', 'polygon']

sys.stdout = Logger_all_print_out_txt(inference_config.log_path_SK)

filelist = filename_out_txt(inference_config.imagelist_SK)
inference_config.display()

################################ Data Loader
class ParkDataset(utils.Dataset):
#데이터셋 가져오는곳.
    def load_park(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        for class_name, class_number in inference_config.names_dict_SK.items():
            self.add_class(inference_config.bbox_polygon_SK, # bbox or polygon
                           class_number, # BlockedDrain
                           class_name) # 1
        
    
        # Train or validation dataset?
        assert subset in ["Training", "Validation", "Test"]
        label_dir = os.path.join(dataset_dir,ARGS.label_folder, subset).replace('\\', '/')
        data_dir = os.path.join(dataset_dir,ARGS.image_folder, subset).replace('\\', '/')
        print(label_dir)
        bbox_file_list = []
        print(datetime.datetime.now(), "check start")


        # ------------------- JSON 파일에서 데이터화 -------------------
    
        # Load annotations
        # json_list = glob.glob(os.path.join(label_dir, "**/*.json").replace('\\', '/'), recursive=True)
        json_list = glob.glob(os.path.join(label_dir, ARGS.dataset_rule).replace('\\', '/'), recursive=True)
        print(os.path.join(label_dir, ARGS.dataset_rule))
        annotations_b = []
        annotations_p = []
        print(len(json_list))
        
        print(datetime.datetime.now(), "list check ok")
        for json_file in tqdm(json_list):
            # ★ SK, 경로 변경
            json_file = json_file.replace('\\', '/')
            
            with open(json_file, 'rb') as f:
                data = json.load(f)
                if 'bbox2d' in data and len(data['bbox2d'])!=0:
                    annotations_b.append([json_file,data])
                    
                # ★SK, polygon
                if 'segmentation' in data and len(data['segmentation'])!=0:
                    annotations_p.append([json_file, data])
                    
        print(datetime.datetime.now(), "list load ok")
        assert (bool(annotations_b) and bool(annotations_p)) == False
        
        # ------------------- (BBOX) self.add_image() -------------------
        # Add images
        for a in tqdm(annotations_b):

            bboxs = [b['bbox'] for b in a[1]['bbox2d']]
            name = [b['name'] for b in a[1]['bbox2d']]
            
            name_dict = inference_config.names_dict_SK

            num_ids = []

            for i,n in enumerate(name) : 
                if n in name_dict :
                    num_ids.append(name_dict[n])
                    
                else : 
                    print(f'{a[0]} 파일 {n} 객체 오류')
                    del bboxs[i]
                    
            # a_img = a[0].split('/')
            # file_name = f'{a_img[-1][:-5]}.jpg'
            # image_path = os.path.join(data_dir,a_img[-5],a_img[-4],a_img[-3],"Camera",file_name).replace('\\', '/')
            image_path = a[0].replace(ARGS.label_folder, ARGS.image_folder)[:-5] + '.jpg'
            file_name = image_path.split('/')[-1]
            if os.path.exists(image_path):
                try:
                    image = skimage.io.imread(image_path)
                    height, width = image.shape[:2]

                    self.add_image(
                        "bbox", # ★ SK 주석만 달았음, image_info['source']
                        image_id='/'.join(image_path.split('/')[-3:]),  # use file name as a unique image id
                        path=image_path,
                        width=width, height=height,
                        bboxs=bboxs,
                        num_ids=num_ids)
                    bbox_file_list.append(image_path)
                    #print(f'{image_path} 불러오기 성공하였슴')
                except :
                    print(f'{image_path} 불러오기 실패')
                    pass
            
            # ★ SK, 이미지 없는 경우 확인
            else:
                return print("이미지가 해당 경로에 없습니다!!", image_path)
                
        # ------------------- (Polygon) self.add_image() -------------------
        for a in tqdm(annotations_p):
            polygons = [p['polygon'] for p in a[1]['segmentation']]
            name = [p['name'] for p in a[1]['segmentation']]
            
            name_dict = inference_config.names_dict_SK
            
            num_ids = []
            
            for i, n in enumerate(name):
                if n in name_dict:
                    num_ids.append(name_dict[n])
                
                else:
                    print(f'{a[0]} 파일 {n} 객체 오류')
                    del polygons[i]
            
            a_img = a[0].split('/')
            file_name = f'{a_img[-1][:-5]}.jpg'

            # a_img = a[0].split('/')
            # file_name = f'{a_img[-1][:-5]}.jpg'
            # image_path = os.path.join(data_dir,a_img[-5],a_img[-4],a_img[-3],"Camera",file_name).replace('\\', '/')
            image_path = a[0].replace(ARGS.label_folder, ARGS.image_folder)[:-5] + '.jpg'
            file_name = image_path.split('/')[-1]
            if os.path.exists(image_path):
                try:
                    image = skimage.io.imread(image_path)
                    height, width = image.shape[:2]
                    
                    self.add_image(
                        "polygon",
                        image_id='/'.join(image_path.split('/')[-3:]),  # use file name as a unique image id
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons,
                        num_ids=num_ids
                    )
                except:
                    print(f'{image_path} 불러오기 실패')
                    pass
        
            else:
                print(image_path)
                print('이미지가 해당 경로에 없어')

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        # ★ SK, 아래랑 똑같아서 주석
        # image_info = self.image_info[image_id]
        # if image_info["source"] != "bbox":
        #     return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        
        if 'bboxs' in info :
            num_ids = info['num_ids']
            mask = np.zeros([info["height"], info["width"], len(info["bboxs"])],
                            dtype=np.uint8)
            
            bboxs =info["bboxs"]

            for i in range(len(bboxs)) :
                bbox = bboxs[i]
                row_s, row_e = int(bbox[1]), int(bbox[3])
                col_s, col_e = int(bbox[0]), int(bbox[2])
                mask[row_s:row_e, col_s:col_e, i] = 1

        if 'polygons' in info:
            num_ids = info['num_ids']
            mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                            dtype=np.uint8)
            
            polygons = info['polygons']
            for i in range(len(polygons)) :
                polygon = polygons[i]
                all_points_y = []
                all_points_x = []
                for po in polygon :
                    x = po[0]
                    y = po[1]
                    all_points_x.append(float(x))
                    all_points_y.append(float(y))
                rr, cc = skimage.draw.polygon(all_points_y,all_points_x)

                rr[rr > mask.shape[0]-1] = mask.shape[0]-1
                cc[cc > mask.shape[1]-1] = mask.shape[1]-1
                
                mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask.astype(np.bool_), num_ids #np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "bbox":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            

################################ Load Dataset
dataset_link = inference_config.dataset_link_SK

dataset_test = ParkDataset()

dataset_test.load_park(dataset_dir=dataset_link, subset = "Test")
dataset_test.prepare() 
print('Test', len(dataset_test.image_ids))


################################ Model Load
model_path = inference_config.model_path_SK
class_names = list(inference_config.names_dict_SK.keys())

test_model = modellib.MaskRCNN(
    mode="inference", 
    config=inference_config,
    model_dir=model_path)

test_model.load_weights(model_path, by_name=True)


################################ Inference

# image_ids = np.random.choice(dataset_test.image_ids, 10)
image_ids = dataset_test.image_ids
예측개수확인클래스리스트 = list(inference_config.names_dict_SK.keys())
예측개수 = {i: 0 for i in 예측개수확인클래스리스트}
예측없음개수 = {i: 0 for i in 예측개수확인클래스리스트}
예측있음개수 = {i: 0 for i in 예측개수확인클래스리스트}
AP0개수 = {i: 0 for i in 예측개수확인클래스리스트}
AP05미만개수 = {i: 0 for i in 예측개수확인클래스리스트}
AP1미만개수 = {i: 0 for i in 예측개수확인클래스리스트}
AP1개수 = {i: 0 for i in 예측개수확인클래스리스트}
클래스별100개모아 = {i: [] for i in 예측개수확인클래스리스트}
names_dict_SK_키밸류거꾸로 = {v: k for k, v in inference_config.names_dict_SK.items()}

if ARGS.shuffle:

    random.shuffle(image_ids)
    print("\n-------- 클래스별 최대 100개씩 랜덤 생성 --------")

    for idx, 클래스이름 in enumerate(클래스별100개모아.keys(), start=1):
        print(f"{idx}/{len(클래스별100개모아.keys())} - {클래스이름}")

        for image_id in tqdm(image_ids):
            
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset_test, inference_config,
                                    image_id)

            try:
                확인클래스 = names_dict_SK_키밸류거꾸로[gt_class_id[0]]

                if 클래스이름 == 확인클래스:
                    클래스별100개모아[클래스이름].append(image_id)
                    image_ids = image_ids[image_ids != image_id]
            except:
                pass
                
            if len(클래스별100개모아[클래스이름]) >= 100:
                break
                

    image_ids = []
    for 클래스이름, image_ids100 in 클래스별100개모아.items():
        print(f"{클래스이름}: {len(image_ids100)} 개")
        image_ids.extend(image_ids100)
    
    random.shuffle(image_ids)

    print("-----------------------------------------------------\n")


if ARGS.bbox_polygon == 'bbox':
    show_mask = False
    show_bbox = True
elif ARGS.bbox_polygon == 'polygon':
    show_mask = True
    show_bbox = False
    
APs = []
sk_continue =0

last_idx = len(image_ids) if inference_config.last_index == None else inference_config.last_index
start_idx = inference_config.start_index
print("start_index:", start_idx)
print("last_index:", last_idx)

for idx, image_id in enumerate(image_ids, start = 1):
    if idx < start_idx:
        continue
    
    if idx > last_idx:
        break

    image_file_name = dataset_test.image_info[image_id]['path'].split('/')[-1]
    
    #이미지 파일명 텍스트 파일에 저장하기
    filelist.write_image_filename(image_file_name)
    if ARGS.detail_number_print:
        print(f"\n■ {idx}/{last_idx}/{len(image_ids)}) {image_file_name}'")
        
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_test, inference_config,
                               image_id)

    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = test_model.detect([image], verbose=1)
    r = results[0]

    if ARGS.individual and len(r["scores"]): # 신뢰도가 가장 높은 결과 한 개 사용 - 1 케이지 1 미꾸리 적용
        max_score_index = np.argmax(r["scores"])

        r["rois"] = np.expand_dims(r["rois"][max_score_index], axis=0)
        r["class_ids"] = np.array([r["class_ids"][max_score_index]])
        r["scores"] = np.array([r["scores"][max_score_index]])
        r['masks'] = np.expand_dims(r['masks'][:, :, max_score_index], axis=-1)


    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'],
                         iou_threshold=ARGS.IoU)
    APs.append(AP)
    print("AP:", AP)
    print('--------')

    예측확인클래스 = names_dict_SK_키밸류거꾸로[gt_class_id[0]]

    예측개수[예측확인클래스] += 1

    if len(r['class_ids']):
        예측있음개수[예측확인클래스] += 1
    else:
        예측없음개수[예측확인클래스] += 1

    if AP == 0:
        AP0개수[예측확인클래스] += 1
    elif 0 < AP < 0.5:
        AP05미만개수[예측확인클래스] += 1
    elif 0.5 <= AP < 1:
        AP1미만개수[예측확인클래스] += 1
    elif AP ==1:
        AP1개수[예측확인클래스] += 1

    if ARGS.individual and ARGS.detail_number_print and (idx % 10 ==  0 or idx == last_idx):
        if idx == last_idx:
            print("\n■ Test Result")

        data = [
            ['Predicted'] + list(예측개수.values()),
            ['Prediction X'] + list(예측없음개수.values()),
            ['Prediction O'] + list(예측있음개수.values()),
            ['AP==0'] + list(AP0개수.values()),
            ['AP < 0.5'] + list(AP05미만개수.values()),
            ['AP < 1'] + list(AP1미만개수.values()),
            ['Ap==1'] + list(AP1개수.values()),
        ]

        headers = ['Number'] + 예측개수확인클래스리스트
        table = tabulate(data, headers=headers, tablefmt='pretty')
        print('\n', table, '\n')
        print('sum AP:', sum(APs))
        print('len AP:', len(APs))
        print(f"mAP: {round(np.mean(APs), 4):.4f}")

    if not ARGS.individual and (idx % 10 ==  0 or idx == last_idx):
        if idx == last_idx:
            print("\n■ Test Result")
        print('\nsum AP:', sum(APs))
        print('len AP:', len(APs))
        print(f"mAP: {round(np.mean(APs), 4):.4f}")
    
    # 먹이생물 show_mask=False, show_bbox=True 수정
    visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id,
                            dataset_test.class_names,
                            figsize=(4, 4), show_mask=show_mask, show_bbox=show_bbox,
                            save_image_path=inference_config.result_save_path_SK + f'/{image_file_name}_1-GT.jpg')
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_test.class_names, r['scores'],
                            figsize=(4, 4), show_mask=show_mask, show_bbox=show_bbox,
                            save_image_path=inference_config.result_save_path_SK + f"/{image_file_name}_2-DT_Prediction O={bool(len(r['class_ids']))}.jpg")

if ARGS.individual and not ARGS.detail_number_print:
    print("\n■ Test Result")
    print('sum AP:', sum(APs))
    print('len AP:', len(APs))
    print(f"mAP: {round(np.mean(APs), 4):.4f}")

# 종료 시간을 한국 시간으로 변환
end_korea_time = datetime.datetime.now(korea_timezone).strftime(date_format)
filelist.write_endtime(end_korea_time)
print(f'{end_korea_time}')
