Попытка создать бота авто-наводчика для игры в Counter-Strike 1.6, но подойдёт под любые игры вообще при нужных весах.

Использование YOLO на изображении взято отсюда: https://github.com/Asadullah-Dal17/yolov4-opencv-python

 Для работы нужны:
- opencv-contrib-python >= 4.5.3.56  (в последней версии что-то переделали и теперь ничего не работает)
- mss
- pyautogui
- numpy

Как работает:
в файле main.py, в monitor выставьте желаемое разрешение,
а в 

with open('CSv3.names', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

net = cv.dnn.readNet('CSv3_6000.weights', 'CSv3.cfg')

поставте везде либо CSv3 либо CSv2. ( CSv2 немного не точный, а CSv3 получился точнее, но он теперь на свои же руки реагирует)

ну и всё получается.

Весы можно скачать здесь: https://drive.google.com/drive/folders/1wy3BOiJsdn_5pdGw8wWbNrmuKWxlfK3b?usp=sharing
