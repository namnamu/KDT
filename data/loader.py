import setting
import tensorflow as tf

import os
class Loader:
    def __init__(self):
        
        ds = tf.keras.utils.image_dataset_from_directory(
            directory=setting.images,
            validation_split=0.2, # 이미지의 80%를 훈련에 사용하고 20%를 유효성 검사에 사용
            subset="both", 
            label_mode='int', # 원핫 인코딩
            seed=123, # 셔플이 기본적으로 참
            batch_size=32, # 일괄처리 묶음. 샘플의 갯수
            image_size=setting.images_size# 이미지 크기
        )
        self.train_ds, self.test_ds=ds

        self.file_list=os.listdir(setting.detail_images) # 사진의 제목들


    def _get_traintest(self):
        return self.train_ds, self.test_ds
    
    def _get_onebyone(self,model):
        self.features = []
        for img_path in self.file_list:
            self.features.append(model.predict(setting.detail_images+img_path))
        return self.features
    
    
