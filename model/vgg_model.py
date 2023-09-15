import setting
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model as kerasModel

from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
class Model:
    def __init__(self):
        model=VGG16(weights='imagenet')
        self.model=kerasModel(
                    inputs=model.input, outputs=model.get_layer("fc1").output
                )
        
    def predict(self,path):
        img=Image.open(path)

        # Resize the image
        img = img.resize(setting.images_size)
        # Convert the image color space
        img = img.convert('RGB')
        # Reformat the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x) #preprocess_input 함수는 모델에 필요한 형식에 이미지를 적절하게 맞추기위한 것. 일부 모델은 0에서 1까지의 값을 가진 이미지를 사용합니다. 다른 모델은 -1에서 +1까지

        # Extract Features
        feature = self.model.predict(x)[0] # 예측

        return feature / np.linalg.norm(feature)
