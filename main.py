
import  xlrd
from splitWord import MySpliter
from MyWord2Ver import MyWord2Ver
from numpy import array
import numpy as np
import pandas as pd
import os

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split#将数据分为测试集和训练集

def getClasslabel100(num):
    return num//100000

def sigmoid(X):
    return 1.0 / (1 + np.exp(-float(X-48127)))

def Z_s(X):
    a=round((float(X)-48127) / 188641 , 3)
    b=1000*a
    b=int(b)
    return b

def test():
    print(getClasslabel100(11200001))


def analysis():
    # 打开文件
    path = os.getcwd()
    excel = xlrd.open_workbook(path+"/sources/BB.xlsx")
    sheet = excel.sheet_by_name("videos")
    #print("总行：" + str(sheet.nrows))
    #print("总列：" + str(sheet.ncols))
    mySpliter = MySpliter()
    myWord2Ver = MyWord2Ver()
    # print(r)

    # 获取词向量
    sentences = []
    inputDic = []
    allWordsList = []
    # 存放播放量
    Y_input = []
    #for rowNum in range(1, sheet.nrows -1 ):
    for rowNum in range(1, 10000):
        sen = sheet.row_values(rowNum, 0, 6)[0]
        play_num = sheet.row_values(rowNum, 0, 6)[5]
        sentence = mySpliter.split(sen )
        # 只有一个词的排除
        if (len(sentence) <= 1):
            continue
        sentences.append(sentence)
        # Y_input.append(getClasslabel(play_num))
        #Y_input.append(getClasslabel100(play_num))
        Y_input.append(Z_s(play_num))
        allWordsList.extend(sentence)
        #print("第"+ str(rowNum) +"行句子分词：" + str(sentence) )
    allWordsSet = set(allWordsList)
    print( "集合中总计单词数目：" +  str(len(allWordsSet)))
    #daochu=pd.DataFrame(data=sentences)
    #daochu.to_csv("/Users/Hewy/Desktop/daochu.csv", encoding='gbk')
    #sentences.to_csv("/Users/Hewy/Desktop/3.csv")

    wordsDimension = 200
    verDic  = myWord2Ver.getVer(sentences, wordsDimension  , 1)
    print( "单词维度数目：" +  str(wordsDimension) )
    #print(verDic)

    X_inputVer = []
    for item in sentences:
        # print(item)
        everySentenceVer = np.empty( 0 )
        row = 0
        for word in item:
            everySentenceVer = np.concatenate( ( everySentenceVer, verDic[word] ), axis = 0 )
            row += 1
        everySentenceVer = everySentenceVer.reshape((row , wordsDimension)).sum(axis=0)
        # cc= everySentenceVer.sum(axis=0)
        X_inputVer.append(everySentenceVer.tolist())
    #print(X_inputVer)

    # print(str(X_inputVer))

    # 划分数据
    X_train,X_test,y_train,y_test=train_test_split(X_inputVer,Y_input,test_size=0.2)
    #利用train_test_split进行将训练集和测试集进行分开，test_size占20%
    # 模型训练
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(150, 100, 50),
                        random_state=1, verbose = False, early_stopping = True,
                        warm_start = True)
    clf.fit(X_train, y_train)
    print("迭代次数: " + str(clf.n_iter_) )
    print("隐层数: " + str(clf.hidden_layer_sizes) )

    # y_pred = clf.predict(X_test)
    score  = clf.score(X_test, y_test)
    print("R值(准确率) = " + str(score) )
     
# predictions = clf.predict(X_test)
# precision, recall, threshold = precision_recall_curve(y_true, y_scores)
# from sklearn.metricsimportclassification_report,confusion_matrix

    X_test1 = input('请输入要分析的主题：')
    c=mySpliter.split(X_test1)
    #print (c)
    d=[c, c]
    #print(d)
    a = []
    a = myWord2Ver.getVer(d, wordsDimension  , 1)
    #print (a)
    b = []
    for item in d:
        # print(item)
        everySentenceVer = np.empty( 0 )
        row = 0
        for word in item:
            everySentenceVer = np.concatenate( ( everySentenceVer, a[word] ), axis = 0 )
            row += 1
        everySentenceVer = everySentenceVer.reshape((row , wordsDimension)).sum(axis=0)
        # cc= everySentenceVer.sum(axis=0)
        b.append(everySentenceVer.tolist())
    #print(b)
    e=b[0]
    f=array(e).reshape(1, -1)
    bfl=clf.predict(f)
    print("该标题估计播放量为" + str(bfl[0]*188+48127) +"次")
    #print("该标题估计播放量为" + str(bfl[0]*100000) +"次")
    
    
    
if __name__ == '__main__':
    analysis()
    # test()




