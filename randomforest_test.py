#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2012/12/21

@author: shakezo_
'''

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

import numpy as np



def feature_sampling(data,feature_num,mtry):

    partial_data = []
    arr = np.arange(feature_num)
    np.random.shuffle(arr)

    for d in data:
        partial_data.append(d[arr[0:mtry]])
    return [partial_data,arr[0:mtry]]



def predict(clf_list,data):
    #�������ɂ�郂�f���̌���
    predict_dic ={}
    for clf in clf_list:
        input = data[clf[1][1]]
        model = clf[0]
        pid =int(model.predict(input)[0])
        predict_dic[pid] = predict_dic.get(pid,0) + 1

    #�������ŃN���X������
    max_count = 0
    max_id =-1
    for k,v in predict_dic.iteritems():
        if v>max_count:
            max_count = v
            max_id = k
    return max_id




if __name__ == '__main__':

    target_names = {}
    #iris�f�[�^�Z�b�g���擾
    iris = load_iris()
    #�^�[�Q�b�g���擾
    for i,name in enumerate(iris.target_names):
        target_names[i] = name


    #�f�[�^����
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)

    #parameter
    tree_num = 500;
    train_num = int(len(x_train)*(2.0/3))
    test_num  = len(x_train)-train_num
    feature_num = len(x_train[0])
    mtry = 2



    #�u�[�g�X�g���b�v�T���v�����O
    data_list = []
    target_list = []
    input_data_list = []
    clf_list = []
    bs = cross_validation.Bootstrap(len(x_train),n_bootstraps=tree_num,train_size=train_num,test_size=test_num, random_state=0)


    #�����_���t�H���X�g�̎��s
    #�g�p��������ʂƃf�[�^���T���v�����O���Č���؂��\�z
    for train_index, test_index in bs:
        data = x_train[train_index]
        target = y_train[train_index]

        data_list.append(data)
        target_list.append(target)

        #�����ʂ̑I��
        input_data = feature_sampling(data,feature_num,mtry)
        input_data_list.append(input_data)

        #����؂̍쐬
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(input_data[0], target)

        #�쐬�����؂ƃf�[�^��ǉ�
        clf_list.append([clf,input_data])


    #�f�[�^�̗\��
    predict_id_list = []
    #test_data_list = iris.data
    correct_num = 0
    for i,data in enumerate(x_test):
        pid=predict(clf_list,data)
        predict_id_list.append(pid)

        if pid == y_test[i]:
            correct_num += 1

    #Accuracy
    print "Accuracy = " ,correct_num/float(len(x_test))

