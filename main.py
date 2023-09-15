import click
from data import Loader
from model import Model,Similarity

@click.command()
@click.option('-f', '--input-img', type = click.STRING, default='', help='비교할 이미지 사진')
def start(input_img=''):
    # print(input_img)
    loader=Loader()
    # train_ds,test_ds=loader.get_traintest()
    # print(len(train_ds),len(test_ds))
    model=Model()
    features=loader._get_onebyone(model)
    query=model.predict(input_img)
    score=Similarity(features,query).get_score()
    print("가장 유사한 순위 Top 10")
    for i in range(len(score)):
        if i>=10:
            break
        print(i+1," - ",loader.file_list[score[i][1]],"의 유사도",score[i][0])
        
if __name__ == "__main__":
    start()