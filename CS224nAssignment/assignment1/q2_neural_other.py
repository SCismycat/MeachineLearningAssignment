#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.10.28 16:47
import numpy as np
from q2_sigmoid import sigmoid,sigmoid_grad
from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive



def forward_backward_prop(data,labels,params,dimensions):
    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
    ##
    hidden = sigmoid(data.dot(W1)+b1)
    out = hidden.dot(W2)+b2 #(H,Dy)
    N = data.shape[0]

    ##反向bp
    y = np.argmax(labels,axis=1)

    cost = 0.0
    corrent_class_score = out[np.arange(N),y].reshape(N,1)#当前输出
    exp_sum = np.sum(np.exp(out),axis=1).reshape(N,1)
    cost += np.sum(np.log(exp_sum)-corrent_class_score)
    cost /= N

    margin = np.exp(out) / exp_sum # softmax函数
    margin[np.arange(N),y] += -1 #(N,Dy)
    margin /= N
    gradW2 = hidden.T.dot(margin) # （H,N）*(N,Dy)
    gradb2 = np.sum(margin,axis=0) # （1，Dy）

    margin1 = margin.dot(W2.T)
    margin1 *= sigmoid_grad(hidden)
    gradW1 = data.T.dot(margin1)
    gradb1 = np.sum(margin1,axis=0)

 ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad

