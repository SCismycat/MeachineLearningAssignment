#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.10.28 11:41
import numpy as np
def softmax(x):
    """Compute the softmax function for each row of the input x.

        It is crucial that this function is optimized for speed because
        it will be used frequently in later code. You might find numpy
        functions np.exp, np.sum, np.reshape, np.max, and numpy
        broadcasting useful for this task.

        Numpy broadcasting documentation:
        http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

        You should also make sure that your code works for a single
        N-dimensional vector (treat the vector as a single row) and
        for M x N matrices. This may be useful for testing later. Also,
        make sure that the dimensions of the output match the input.

        You must implement the optimization in problem 1(a) of the
        written assignment!

        Arguments:
        x -- A N dimensional vector or M x N dimensional numpy matrix.

        Return:
        x -- You are allowed to modify x in-place
        """
    orig_shape = x.shape
    if len(x.shape)>1:
        # Matrix
        ###your code here
        exp_minmax = lambda x:np.exp(x-np.max(x))
        denom = lambda  x: 1.0/sum(x)
        """
    import numpy as np
    def f(a):
        return (a[0]+a[1])*2
    b=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    np.apply_along_axis(f,0,b) 
    #结果:array([12, 16, 20, 24])
    #(1+5)*2=12  (2+6)*2=16依次类推
    np.apply_along_axis(f,1,b)
    #结果:array([ 6, 22, 38])
    #(1+2)*2=6  (5+6)*2=22依次类推
        """
        x = np.apply_along_axis(exp_minmax,1,x)
        denominator = np.apply_along_axis(denom,1,x)
        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0],1))
        x = x * denominator
    else:
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0/np.sum(numerator)

        x = numerator.dot(denominator)
    assert x.shape == orig_shape
    return x

def test_softmax_basic():
    print("runing basic tests..")
    test1 = softmax(np.array([1,2]))
    print(test1)

    ans1 = np.array([0.26894142, 0.73105858])
    assert np.allclose(test1,ans1,rtol=1e-05,atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print(test2)
