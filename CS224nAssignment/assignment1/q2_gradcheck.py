#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Leslee
# @Email    : leelovesc@gmail.com
# @Time    : 2019.10.28 14:40
# 采用数值计算方法对公式计算梯度，和正式的梯度进行对比
import random
import numpy as np
def gradcheck_naive(f,x):
    """ Gradient check for a function f.

        Arguments:
        f -- a function that takes a single argument and outputs the
             cost and its gradients
        x -- the point (numpy array) to check the gradient at
        """
    # 检查随机数实现的内部状态， 保证跨实现产生相同随机数？
    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x)
    h = 1e-4

    it = np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        # Try modifying x[ix] with h defined above to compute
        # numerical gradients. Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.
        ### YOUR CODE HERE
        # 求f(x+x0)
        x[ix] += h
        random.setstate(rndstate)
        new_f1 = f(x)[0]
        # 求f(x-x0)
        x[ix] -= 2*h
        random.setstate(rndstate)
        new_f2 = f(x)[0]
        # 还原原来的数值
        x[ix] += h

        numgrad = (new_f1 - new_f2) / (2*h)

        # 比较梯度
        reldiff = abs(numgrad - grad[ix]) / max(1,abs(numgrad),abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad))
            return
        it.iternext()
    print("Gradient check passed!")

def sanity_check():
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")

    gradcheck_naive(quad, np.array(123.456))  # scalar test
    gradcheck_naive(quad, np.random.randn(3, ))  # 1-D test
    gradcheck_naive(quad, np.random.randn(4, 5))  # 2-D test
    print("")


