from timeit import Timer
import scipy
import numpy
import math

n = 100

# test the time for the factorial function obtained in different ways:

if __name__ == '__main__':

    setupstr="""
import scipy, numpy, math
n = 100
"""

    method2="""
scipy.math.factorial(n)
"""

    method3="""
numpy.math.factorial(n)
"""

    method4="""
math.factorial(n)
"""

    nl = 1000
    t2 = Timer(method2, setupstr).timeit(nl)
    t3 = Timer(method3, setupstr).timeit(nl)
    t4 = Timer(method4, setupstr).timeit(nl)


    print 'method2', t2
    print 'method3', t3
    print 'method4', t4


    print scipy.math.factorial(n)
    print numpy.math.factorial(n)
    print math.factorial(n)
