################################ Args
import argparse
import ast
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--classes_names_dict', type=str, help='classes_names_dict')
parser.add_argument('--name', type=str, help='model name')
parser.add_argument('--dataset', type=str, help='dataset path')
parser.add_argument('--dataset_rule', type=str, help='dataset_path_rule over TVT', default="**/*.json")
parser.add_argument('--exclude_rule', type=str, help='exclude dataset_path_rule over TVT')
parser.add_argument('--epoch', type=int, help='epoch')
parser.add_argument('--early_stop', type=int, help='use early_stop')
parser.add_argument('--bbox_polygon', type=str, help='bbox or polygon')
parser.add_argument('--weights', type=str, help='model path')
parser.add_argument('--save_dir', type=str, help='model save_dir')
parser.add_argument('--STEPS_PER_EPOCH', type=int, help='model STEPS_PER_EPOCH', default=100)
ARGS = parser.parse_args()

################################ Logger
class Logger_all_print_out_txt:
 
    def __init__(self, log_path):
        self.log_path = log_path
        self.console = sys.stdout

        # 시작 시간을 한국 시간으로 변환
        start_korea_time = datetime.datetime.now(korea_timezone).strftime(date_format)

        
        with open(self.log_path, 'w', encoding='utf-8 sig') as f:
            f.write(f"{start_korea_time}\n\n")
            f.write(f'''!source /content/venv/bin/activate; python train.py\\''')
            f.write(f'''\n--classes_names_dict "{ARGS.classes_names_dict}"\\''')
            f.write(f'''\n--name "{ARGS.name}"\\''')
            f.write(f'''\n--dataset "{ARGS.dataset}"\\''')
            f.write(f'''\n--epoch {ARGS.epoch}\\''')
            f.write(f'''\n--early_stop {ARGS.early_stop}\\''')
            f.write(f'''\n--bbox_polygon "{ARGS.bbox_polygon}"\\''')
            f.write(f'''\n--weights "{ARGS.weights}"\\''')
            f.write(f'''\n--save_dir "{ARGS.save_dir}"\n\n''')
            pass
 
    def write(self, message):
        self.console.write(message)
        self.write_onlytxt(message)
        
    def write_onlytxt(self, message):
        with open(self.log_path, 'a', encoding='utf-8 sig') as f:
            f.write(message)
 
    def flush(self):
        self.console.flush()

################################ Path

import os
import json
import numpy as np
from PIL import Image, ImageDraw
import time

from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize

print(ARGS.early_stop, type(ARGS.early_stop))
if not ARGS.early_stop:
    print("--- early_stop 을 사용하지 않습니다..")
    import mrcnn.model_not_early_stopping as modellib # ★ SK, E-S option X
else:
    print("--- early_stop 을 사용합니다 !!")
    import mrcnn.model as modellib # ★ SK, E-S option

import datetime
import pytz
import glob
import skimage.draw
from tqdm import tqdm

# MODEL_DIR = ARGS.save_dir
# 한국 시간대 설정
korea_timezone = pytz.timezone('Asia/Seoul')
date_format = "%Y-%m-%d %H:%M:%S %Z%z"
# 코드 시작 시간 기록
start_time = time.time()

################################ Config
class ParkConfig(Config):
    names_dict_SK = ast.literal_eval(ARGS.classes_names_dict)

    NAME = ARGS.name

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 1 + len(names_dict_SK)  # BG + classes

    STEPS_PER_EPOCH = ARGS.STEPS_PER_EPOCH
    DETECTION_MIN_CONFIDENCE = 0.9

    dataset_link_SK = ARGS.dataset

    if ARGS.weights.upper() == 'COCO':
        pretrained_model_dir_SK = 'mask_rcnn_coco.h5'
    else:
        pretrained_model_dir_SK = ARGS.weights

    train_epoch_SK = ARGS.epoch
    
    bbox_polygon_SK =ARGS.bbox_polygon

    model_save_path_SK = ARGS.save_dir
    if not os.path.exists(model_save_path_SK):
        os.makedirs(model_save_path_SK)
    else:
        # raise Exception(f"{model_save_path_SK} is already exists")
        pass

    log_path_SK = ARGS.save_dir + '/log.txt'
    assert not os.path.exists(log_path_SK)
    
parkConfig = ParkConfig()

assert parkConfig.bbox_polygon_SK in ['bbox', 'polygon']

sys.stdout = Logger_all_print_out_txt(parkConfig.log_path_SK)
parkConfig.display()

################################ Data Loader
class ParkDataset(utils.Dataset):
#데이터셋 가져오는곳.
    def load_park(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        for class_name, class_number in parkConfig.names_dict_SK.items():
            self.add_class(parkConfig.bbox_polygon_SK, # bbox or polygon
                           class_number, # BlockedDrain
                           class_name) # 1
        
    
        # Train or validation dataset?
        assert subset in ["Training", "Validation", "Test"]
        label_dir = os.path.join(dataset_dir,"2.라벨링데이터", subset).replace('\\', '/')
        data_dir = os.path.join(dataset_dir,"1.원천데이터", subset).replace('\\', '/')
        print(label_dir)
        bbox_file_list = []
        print(datetime.datetime.now(), "check start")


        # ------------------- JSON 파일에서 데이터화 -------------------
    
        # Load annotations
        # json_list = glob.glob(os.path.join(label_dir, "**/*.json").replace('\\', '/'), recursive=True)
        json_list = glob.glob(os.path.join(label_dir, ARGS.dataset_rule).replace('\\', '/'), recursive=True)
        if ARGS.exclude_rule:
            exclude_json_list = glob.glob(os.path.join(label_dir, ARGS.exclude_rule).replace('\\', '/'), recursive=True)
            for exclude_json in exclude_json_list:
                json_list.remove(exclude_json)

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
            
            name_dict = parkConfig.names_dict_SK

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
            image_path = a[0].replace('2.라벨링데이터', '1.원천데이터')[:-5] + '.jpg'
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
                except :
                    pass

                # ------------------- (Polygon) self.add_image() -------------------
        for SK_idx, a in enumerate(tqdm(annotations_p)):
            polygons = [p['polygon'] for p in a[1]['segmentation']]
            name = [p['name'] for p in a[1]['segmentation']]
            
            name_dict = parkConfig.names_dict_SK
            
            num_ids = []
            
            for i, n in enumerate(name):
                if n in name_dict:
                    num_ids.append(name_dict[n])
                
                else:
                    del polygons[i]
            
            a_img = a[0].split('/')
            file_name = f'{a_img[-1][:-5]}.jpg'

            # a_img = a[0].split('/')
            # file_name = f'{a_img[-1][:-5]}.jpg'
            # image_path = os.path.join(data_dir,a_img[-5],a_img[-4],a_img[-3],"Camera",file_name).replace('\\', '/')
            image_path = a[0].replace('2.라벨링데이터', '1.원천데이터')[:-5] + '.jpg'
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
                    pass

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
dataset_train = ParkDataset()

dataset_link = parkConfig.dataset_link_SK

dataset_train.load_park(dataset_dir=dataset_link, subset = "Training")
dataset_train.prepare()

training_dir = os.path.join(dataset_link,"2.라벨링데이터", "Training").replace('\\', '/')
training_list = glob.glob(os.path.join(training_dir, ARGS.dataset_rule).replace('\\', '/'), recursive=True)
print('Train', len(training_list))

dataset_val = ParkDataset()
dataset_val.load_park(dataset_dir=dataset_link, subset = "Validation")
dataset_val.prepare()

validation_dir = os.path.join(dataset_link,"2.라벨링데이터", "Validation").replace('\\', '/')
validation_list = glob.glob(os.path.join(validation_dir, ARGS.dataset_rule).replace('\\', '/'), recursive=True)
print('Validation', len(validation_list))

################################ Datset Test
image_ids = np.random.choice(dataset_train.image_ids, 30)

for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks_SK(image, mask, class_ids, dataset_train.class_names, limit=2,
                                                    image_file_name=dataset_train.image_info[image_id]['path'].split('/')[-1])
    if np.all(mask == False):
        print('★★★★★★★★★★★★★마스크없음★★★★★★★★★★★★★')
        print(dataset_train.image_info[image_id]['path'])
        print()

################################ Model Load
model = modellib.MaskRCNN(
    mode="training",
    config=parkConfig,
    model_dir=parkConfig.model_save_path_SK)

if ARGS.weights.upper() == 'COCO':
    model.load_weights(filepath=parkConfig.pretrained_model_dir_SK, 
                    by_name=True,
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
else:
    model.load_weights(parkConfig.pretrained_model_dir_SK, by_name=True)
print('사전학습모델불러오기완료')

############################### Model Train
model.train(
    dataset_train,
    dataset_val, 
    learning_rate=parkConfig.LEARNING_RATE,
    epochs=parkConfig.train_epoch_SK,
    layers="all")

# 종료 시간을 한국 시간으로 변환
end_korea_time = datetime.datetime.now(korea_timezone).strftime(date_format)

print(f'{end_korea_time}')