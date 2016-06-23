import numpy as np
from transformations.s_vf import SVF
from matplotlib import pyplot as plt
from visualizer.fields_at_the_window import see_field

"""
Module aimed to the investigation of the nature of the
random generated svf. What are the parameters that make the svf still
an svf?
"""

shape = (20, 20, 1, 1, 2)

sigma_init = 4
sigma_gaussian_filter = 2


svf_im0   = SVF.generate_random_smooth(shape=shape,
                                         sigma=sigma_init,
                                         sigma_gaussian_filter=sigma_gaussian_filter)


see_field(svf_im0)


print 'min def0 ' + str(np.min(svf_im0.field))
print 'max def0 ' + str(np.max(svf_im0.field))
print 'median def0 ' + str(np.median(svf_im0.field))
print 'Norm: ' + str(svf_im0.norm(normalized=True))

plt.show()


