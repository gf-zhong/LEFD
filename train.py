import warnings
from ultralytics import YOLO
warnings.filterwarnings('ignore')
if __name__ == '__main__':
    model = YOLO(r'')
    #model.load('')  # loading pretrain weights
    model.train(data=r'',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=1,
                close_mosaic=10,
                workers=4,
                device='0',
                optimizer='SGD',  # using SGD
                #resume=r'',  # last.pt path
                amp=False,  # close amp
                # fraction=0.2,
                project='runs/train',
                name='',
                )
