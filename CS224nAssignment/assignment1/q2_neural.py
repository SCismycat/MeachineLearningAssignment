#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.10.28 15:28
import numpy as np
from q2_sigmoid import sigmoid
from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
import random


def forword_backward_prop(data,labels,params,dimensions):
    """
        Forward and backward propagation for a two-layer sigmoidal network

        Compute the forward propagation and for the cross entropy cost,
        and backward propagation for the gradients for all parameters.

        Arguments:
        data -- M x Dx matrix, where each row is a training example.
        labels -- M x Dy matrix, where each row is a one-hot vector.
        params -- Model parameters, these are unpacked for you.
        dimensions -- A tuple of input dimension, number of hidden units
                      and output dimension
        """
    # 解压模型参数
    ofs = 0
    Dx,H,Dy = (dimensions[0],dimensions[1],dimensions[2])
    W1 = np.reshape(params[ofs:ofs+Dx*H],(Dx, H))
    ofs += Dx *H
    b1 = np.reshape(params[ofs:ofs+H], (1,H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs+H*Dy],(H,Dy))
    ofs += H*Dy
    b2 = np.reshape(params[ofs:ofs+Dy],[1, Dy])
    ### YOUR CODE HERE:前向传播
    h = sigmoid(np.dot(data,W1)+b1)
    yhat = softmax(np.dot(h,W2)+b2)
    ## end code

    ### YOUR CODE HERE :反向传播
    cost = np.sum(-np.log(yhat[labels==1]))/data.shape[0]

    d3 = (yhat -labels) / data.shape[0] # 对应y^-y,是输出层对y的求导
    gradW2 = np.dot(h.T,d3) #向前传递到隐藏层的梯度更新值
    gradb2 = np.sum(d3,0,keepdims=True)# b是一个batch样本的平均

    dh = np.dot(d3,W2.T) # 隐藏层的梯度
    grad_h = sigmoid(h) * dh # 隐藏层的梯度

    gradW1 = np.dot(data.T,grad_h) # 向前传递到输入层的梯度更新值
    gradb1 = np.sum(grad_h,0) # 平均

    grad = np.concatenate((gradW1.flatten(),gradb1.flatten(),
                           gradW2.flatten(),gradb2.flatten()))
    return cost,grad

def sanity_check():
    print("running sanity check..")
    N = 20
    dimessions = [10,5,10]
    data = np.random.randn(N,dimessions[0])
    labels = np.zeros((N,dimessions[2]))
    for i in range(N):
        labels[i,random.randint(0,dimessions[2]-1)] = 1

    params = np.random.randn((dimessions[0]+1)*dimessions[1]+(dimessions[1]+1)*dimessions[2],)

    gradcheck_naive(lambda params:forword_backward_prop(data,labels,params,dimessions),params)

sanity_check()