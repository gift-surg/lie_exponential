import numpy as np


"""
Module to explore einsum and tensordot.
"""

#a = np.arange(60.).reshape([3, 4, 5])
#b = np.arange(24.).reshape([4, 3, 2])
#c = np.einsum('ijk,ijl->ik', a, b)

#print a
#print b
#print c

#print c.shape




a = np.arange(16*3).reshape([4, 4, 3])
b = np.tile(a, (1, 1, 3))
print a
print ''
print b



print ''

c = np.sum(b.reshape([4, 4, 3, 3]), axis=3).reshape([4, 4, 3])

print c

a[2,2,:] = [1,2,3]

print a[2,2,:]
