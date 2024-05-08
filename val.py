import warnings
from ultralytics import YOLO

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    model = YOLO('')
    model.val(data=r'',
              split='test',
              imgsz=640,
              batch=32,
              rect=False,
              save_json=True,  # if you need to cal coco metrice
              project='runs/val',
              name='',
              )
