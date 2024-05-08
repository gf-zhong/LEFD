import warnings
from ultralytics import YOLO

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    model = YOLO(r'')  # select your model.pt path
    model.predict(source=r'',
                  imgsz=640,
                  project='runs/detect',
                  name='',
                  save=True,
                  #visualize=True
                  )
