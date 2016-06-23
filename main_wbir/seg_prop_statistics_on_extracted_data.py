import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


from utils.path_manager_3 import full_path_roi_indexes, full_path_means_dice_no_aei, full_path_means_dice_aei, \
    full_path_labelled_dice_no_aei, full_path_labelled_dice_aei
# full_path_means_times_no_aei, full_path_means_times_aei,
#  full_path_labelled_times_no_aei, full_path_labelled_times_aei

"""
Phase 3: statistics are preformed on the extracted arrays
Input: arrays where the data from the cluster have been converted and stored.
Output: statistics and graphs.
"""

roi_indexes = np.load(full_path_roi_indexes)

means_dice_no_aei    = np.load(full_path_means_dice_no_aei)
means_dice_aei       = np.load(full_path_means_dice_aei)
labelled_dice_no_aei = np.load(full_path_labelled_dice_no_aei)
labelled_dice_aei    = np.load(full_path_labelled_dice_aei)

# Visual assessment to validate that the data are correct:

if 0:
    print 'aei'

    print means_dice_aei[:7, :7]

    print 'no aei'

    print means_dice_no_aei[:7, :7]

    print 'aei labelled zone 3 and -1 '

    f = 0  # multiple of 7 between 0 and 28
    for j in range(138):
        print
        print 'slice ' + str(j)
        print labelled_dice_aei[f:(f+7), f:(f+7), j]
        print

    print 'no aei labelled zone 3 and -1 '

    print labelled_dice_no_aei[:7, :7, 3]
    print labelled_dice_no_aei[:7, :7, -1]

    print
    print 'shapes'
    print roi_indexes.shape
    print means_dice_no_aei.shape
    print means_dice_aei.shape
    print labelled_dice_no_aei.shape
    print labelled_dice_aei.shape


## Check if the means are really the means:

cont = 0

if 0:
    for i in range(35):
        for j in range(35):
            print
            print i, j
            print np.round(np.mean(labelled_dice_no_aei[i, j, :]), decimals=5)
            print np.round(means_dice_no_aei[i, j], decimals=5)
            if np.round(np.mean(labelled_dice_no_aei[i, j, :]), decimals=5) != np.round(means_dice_no_aei[i, j], decimals=5):
                cont += 1
                #print
                print 'means not coherent ref ' + str(i) + ' flo '  + str(j)
                #print

if 1:
    np.testing.assert_array_almost_equal(np.mean(labelled_dice_no_aei, axis=2), means_dice_no_aei, decimal=5)
    np.testing.assert_array_almost_equal(np.mean(labelled_dice_aei, axis=2), means_dice_aei, decimal=5)

# Elaborate statistics on data:
# vectorise upper and lower triangles of the matrices

num_elements = labelled_dice_no_aei.shape[0]

# lower triangle no aei
means_dice_def_on_disp_no_aei = np.array([means_dice_no_aei[i, j] for i in range(1, num_elements)
                                                                  for j in range(0, i)])


# upper triangle no aei
means_dice_disp_on_def_no_aei = np.array([means_dice_no_aei[i, j] for i in range(num_elements)
                                                                  for j in range(i+1, num_elements)])
# lower triangle no aei
means_dice_def_on_disp_aei = np.array([means_dice_aei[i, j] for i in range(1, num_elements)
                                                            for j in range(0, i)])
# upper triangle aei
means_dice_disp_on_def_aei = np.array([means_dice_aei[i, j] for i in range(num_elements)
                                                                  for j in range(i+1, num_elements)])

# mean of means


#### UPPER

mu_up_no_aei  = np.mean(means_dice_disp_on_def_no_aei)
std_up_no_aei =  np.std(means_dice_disp_on_def_no_aei)

mu_up_aei  = np.mean(means_dice_disp_on_def_aei)
std_up_aei = np.std(means_dice_disp_on_def_aei)

### LOWER

mu_low_no_aei = np.mean(means_dice_def_on_disp_no_aei)
std_low_no_aei = np.std(means_dice_def_on_disp_no_aei)

mu_low_aei = np.mean(means_dice_def_on_disp_aei)
std_low_aei = np.std(means_dice_def_on_disp_aei)


print 'upper no aei: mean, stdev = '
print mu_up_no_aei, std_up_no_aei
print 'upper aei: mean, stdev = '
print mu_up_aei, std_up_aei
print
print 'lower no aei: mean, stdev = '
print mu_low_no_aei, std_low_no_aei
print 'lower no aei: mean, stdev = '
print mu_low_aei, std_low_aei

print
# t-test UPPER:
print 't-test upper = '
print stats.ttest_ind(means_dice_def_on_disp_no_aei, means_dice_def_on_disp_aei)

# t-test LOWER:
print 't-test lower = '
print stats.ttest_ind(means_dice_disp_on_def_no_aei, means_dice_disp_on_def_aei)

print
print 'H_0 must be accepted. The difference in the dice score is not statistically significant.'


# Figure 1: boxplot

plt.figure()
plt.boxplot([means_dice_disp_on_def_no_aei, means_dice_disp_on_def_aei])
plt.grid()
plt.show()



