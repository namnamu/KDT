# KDT 4기
Tensorflow를 활용한 딥러닝 모델 개발 실습
 ## 이미지 유사도 구현

- 주제 : 사전 학습된 keras 속의 CNN 모델을 이용하여 포켓몬의 유사도 구하기

실행방법   

    python .\main.py -f "대상 사진 파일 위치"


setting 파일 변경   

    images=detail_images의 상위 폴더   
    detail_images=학습용 사진 데이터가 있는 디렉토리   
    images_size=사진 크기 튜플

폴더 구성

    data
    |--__init.py__
    |--loader.py    : 데이터 로드
    model
    |--__init.py__
    |--simiarity.py : 유사도 구하는 공식이 모여있다.
    |--vgg_model.py : vgg16모델 기능을 사용한다.
    main.py         : 시작 파일. 모델과 데이터를 활용한 실질적인 동작
    setting.py      : 개인 환경 세팅 파일


