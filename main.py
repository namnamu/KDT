# import click
from data import Loader
from model import Model,Similarity

# @click.command()
# @click.option('-f', '--input-img', type = click.STRING, default='', help='비교할 이미지 사진')
def start(input_img=''):
    # print(input_img)
    loader=Loader()
    # train_ds,test_ds=loader.get_traintest()
    # print(len(train_ds),len(test_ds))
    model=Model()
    features=loader._get_onebyone(model)
    query=model.predict("C:/Users/skadl/OneDrive/Desktop/ai/perceptron_cnn_rnn/poke_img/images/images/abra.png")
    score=Similarity(features,query).get_score()
    print("가장 유사한 순위 Top 10")
    for i in range(len(score)):
        print(i," - ",loader.file_list[score[i][1]],"의 유사도",score[i][0])
        if i>=10:
            break
if __name__ == "__main__":
    start()