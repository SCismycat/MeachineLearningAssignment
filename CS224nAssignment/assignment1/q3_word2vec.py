#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.10.30 10:41
import random

import numpy as np
from q1_softmax import *
from q2_sigmoid import *
from q2_gradcheck import *

def normalizeRows(x):
    """ Row normalization function

        Implement a function that normalizes each row of a matrix to have
        unit length.
        """
    denom = np.apply_along_axis(lambda x:np.sqrt(x.T.dot(x)),1,x)
    x /= denom[:,None]
    return x

def text_normalize_rows():
    print("Start Testing")
    x = normalizeRows(np.array([[3.0,4.0],[1,2]]))
    print(x)

def softmaxCostAndGradient(predicted,target,outputVectors,dataset):
    """ Softmax cost function for word2vec models
        word2vec的softmax的实现方式
        Implement the cost and gradients for one predicted word vector
        and one target word vector as a building block for word2vec
        models, assuming the softmax prediction function and cross
        entropy loss.

        Arguments:
        predicted -- numpy ndarray, predicted word vector (\hat{v} in
                     the written component)
        target -- integer, the index of the target word
        outputVectors -- "output" vectors (as rows) for all tokens
        dataset -- needed for negative sampling, unused here.

        Return:
        cost -- cross entropy cost for the softmax word prediction
        gradPred -- the gradient with respect to the predicted word
               vector
        grad -- the gradient with respect to all the other word
               vectors

        We will not provide starter code for this function, but feel
        free to reference the code you previously wrote for this
        assignment!
        """
    # 计算预测值
    vhat = predicted
    z = np.dot(outputVectors,vhat)
    preds = softmax(z)

    # 计算损失
    cost = -np.log(preds[target])
    # 梯度
    z = preds.copy()
    z[target] -= 1.0

    grad = np.outer(z,vhat)
    gradPred = np.dot(outputVectors.T,z)
    return cost,gradPred,grad

def getNegativeSamples(target,dataset,K):
    """Samples K indexes which are not the target"""
    indices = [None] * K
    for k in range(K):
        newIdx = dataset.sampleTokenIdx()
        while newIdx == target:
            newIdx = dataset.sampleTokenIdx()
        indices[k] = newIdx
    return indices

def negSamplingCostAndGradient(predicted,target,outputVectors,dataset,K=10):
    """ Negative sampling cost function for word2vec models

        Implement the cost and gradients for one predicted word vector
        and one target word vector as a building block for word2vec
        models, using the negative sampling technique. K is the sample
        size.
        predicted -- numpy ndarray, predicted word vector (\hat{v} in
                     the written component)
        target -- integer, the index of the target word
        outputVectors -- "output" vectors (as rows) for all tokens
        dataset -- needed for negative sampling, unused here.
        Note: See test_word2vec below for dataset's initialization.

        Arguments/Return Specifications: same as softmaxCostAndGradient
        """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target,dataset,K))
    ## YOUR CODE HERE
    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)

    cost = 0
    z = sigmoid(np.dot(outputVectors[target],predicted))

    cost -= np.log(z)
    grad[target] += predicted * (z-1.0)
    gradPred += outputVectors[target] * (z-1.0)

    for k in range(K):
        samp = indices[k+1]
        z = sigmoid(np.dot(outputVectors[samp],predicted))
        cost -= np.log(1.0-z)
        grad[samp] += predicted * z
        gradPred += outputVectors[samp] * z

    return cost,gradPred,grad

def skipgram(currentWord,C,contextWords,tokens,inputVectors,outputVectors,
             dataset,word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

       Implement the skip-gram model in this function.

       Arguments:
       currrentWord -- a string of the current center word
       C -- integer, context size
       contextWords -- list of no more than 2*C strings, the context words
       tokens -- a dictionary that maps words to their indices in
                 the word vector list
       inputVectors -- "input" word vectors (as rows) for all tokens
       outputVectors -- "output" word vectors (as rows) for all tokens
       word2vecCostAndGradient -- the cost and gradient function for
                                  a prediction vector given the target
                                  word vectors, could be one of the two
                                  cost functions you implemented above.

       Return:
       cost -- the cost function value for the skip-gram model
       grad -- the gradient with respect to the word vectors
       """
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    cword_idx = tokens[currentWord]
    vhat = inputVectors[cword_idx]

    for j in contextWords:
        u_idx = tokens[j]
        c_cost,c_grad_in,c_grad_out =\
            word2vecCostAndGradient(vhat,u_idx,outputVectors,dataset)
        cost += c_cost
        gradIn[cword_idx] += c_grad_in
        gradOut += c_grad_out

    return cost, gradIn, gradOut

def cbow(currentWord,C,contextWords,tokens,inputVectors,outputVectors,
         dataset,word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

        Implement the continuous bag-of-words model in this function.

        Arguments/Return specifications: same as the skip-gram model

        Extra credit: Implementing CBOW is optional, but the gradient
        derivations are not. If you decide not to implement CBOW, remove
        the NotImplementedError.
        """
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ##
    predicted_indices = [tokens[word] for word in contextWords]
    predicted_vectors = inputVectors[predicted_indices]
    predicted = np.sum(predicted_vectors,axis=0)
    target = tokens[currentWord]
    cost, gradIn_predicted, gradOut = word2vecCostAndGradient(predicted,target,outputVectors,dataset)
    for i in predicted_indices:
        gradIn[i] += gradIn_predicted
    return cost, gradIn, gradOut

def word2vec_sgd_wrapper(word2vecModel,tokens,wordVectors,dataset,C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = int(wordVectors.shape[0])
    inputVectors = wordVectors[:N // 2,:]
    outputVectors = wordVectors[N//2:,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c,gin,gout = word2vecModel(centerword,C1,context,tokens,inputVectors,outputVectors,
                                   dataset,word2vecCostAndGradient)
        cost += c/batchsize/denom
        grad[:N//2,:] += gin/batchsize/denom
        grad[N//2:,:] += gout/batchsize/denom
    return cost,grad

def test_word2vec():
    dataset = type('dummy',(),{})()

    def dummySampleTokenIdx():
        return random.randint(0,4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)],[tokens[random.randint(0,4)] for i in range(2*C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9268)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
    print("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
                    dummy_vectors)
    print("\n==== Gradient check for CBOW      ====")

    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
                    dummy_vectors)

    print("\n=====result =======")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
             dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset))

    print(skipgram("c", 1, ["a", "b"],
             dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
             negSamplingCostAndGradient))

    print(cbow("a", 2, ["a", "b", "c", "a"],
         dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset))

    print(cbow("a", 2, ["a", "b", "a", "c"],
         dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
         negSamplingCostAndGradient))

if __name__ == '__main__':
    test_word2vec()



