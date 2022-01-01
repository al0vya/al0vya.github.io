#!/usr/bin/env python
# coding: utf-8

# # Mean absolute deviation (MAD)
# 
# ## Why use MAD: efficiency
# 
# The mean absolute deviation MAD is a more intuitive measure of deviation than the standard deviation STD as follows:
# 
# $\text{MAD} = \frac{1}{n} \sum_{i=1}^{n} \lvert x_i - \overline{x} \rvert$.
# 
# MAD is more "efficient" than STD. Efficient here means that MAD varies less than STD. For example, say we take N samples and compute the MAD and STD. We can take N samples M times and compute the MAD and STD each time. We would end up with an array of MADs and STDs of length M.
# 
# <pre>
# 1 : (1, 2, 3, ... N) -> compute -> (MAD) (STD)
# 2 : (1, 2, 3, ... N) -> compute -> (MAD) (STD)  
# . .          .             .         .     .
# . .          .             .         .     .
# . .          .             .         .     .
# M : (1, 2, 3, ... N) -> compute -> (MAD) (STD)
# </pre>
# 
# Then we can calculate the variance of MAD and STD. The lower the variance the better, as a lower variance implies the computed MAD or STD varies less across the M sample sets.
# 
# Consider taking a ratio between the variances of the STDs vs the MADs:
# 
# $\text{efficiency} = \frac{\text{variance}_{STD}}{\text{variance}_{MAD}}$,
# 
# where $\text{efficiency} > 1$ implies MAD is more efficient.
# 
# The efficiency is computed using
# 
# $\text{efficiency} = \frac{\text{variance}_{STD}}{\text{mean}_{STD}} / \frac{\text{variance}_{MAD}}{\text{mean}_{MAD}}$,
# 
# where the means are used to normalise the variances.
# 

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.pylab  as pylab  

import numpy as np
import scipy.stats

from mpl_toolkits.mplot3d import Axes3D

params = {
    "legend.fontsize" : "xx-large",
    "axes.labelsize"  : "xx-large",
    "axes.titlesize"  : "xx-large",
    "xtick.labelsize" : "xx-large",
    "ytick.labelsize" : "xx-large"
}

pylab.rcParams.update(params)


# In[2]:


def mad(data):
    mean = np.mean(data)
    
    absolute_deviations = np.abs(data - mean)
    
    return np.mean(absolute_deviations)

def get_efficiency(data):
    
    return np.var(data) / (np.mean(data) ** 2)

def two_norms(
        num_samples,
        a,           # parameter controlling STD of high STD normal distribution
        p            # sampling frequency from high STD normal distribution
    ):
        """
        This function samples from two different normal distributions, one with STD = 1 and
        another with STD = sqrt(1 + a).
        """
        
        mean = 0
        
        sigma_norm  = 1
        sigma_spike = 1 + a
        
        X_norm  = scipy.stats.norm.rvs(loc=mean, scale=sigma_norm,  size=num_samples)
        X_spike = scipy.stats.norm.rvs(loc=mean, scale=sigma_spike, size=num_samples)
        
        i = scipy.stats.binom.rvs(n=1, p=(1-p), size=num_samples)
        
        X = i * X_norm + (1 - i) * X_spike
        
        return X
    
def get_efficiency_sample(
        a=0, # assume purely Gaussian case by default
        p=0  # never sample from high STD distribution by default
    ):
    """
    When using our samples, the MAD and STD can respectively differ depending on the samples.
    
    That is, the MAD/STD of our samples will differ each time we compute them.
    
    Therefore, we need to compute an array of MADs/STDs using several sets of samples.
    """
    
    num_samples         = 1_000 # number of samples per set
    num_sets_of_samples = 10_000  # number of sets of samples
    
    sets_of_samples = [two_norms(num_samples, a, p) for sample_set in range(num_sets_of_samples)]
    
    stds = [scipy.stats.tstd(sample_set) for sample_set in sets_of_samples]
    mads = [mad(sample_set)              for sample_set in sets_of_samples]
    
    std_efficiency = get_efficiency(stds)
    mad_efficiency = get_efficiency(mads)
    
    return std_efficiency / mad_efficiency


# ## Efficiency for a normal distribution
# 
# The efficiency has been analytically found for the Gaussian case (normal distribution) to be 0.875. This efficiency should be replicated in the function `get_efficiency_sample` using the default parameters.

# In[3]:


print(
    "Analytical relative efficiency STD / MAD: 0.875\n" +
    "Sampled relative efficiency STD / MAD: %0.5s" % get_efficiency_sample()
)


# ## Efficiency for progressively fatter tail distributions
# 
# Next, we would like to see how the efficiency changes when we sample from fatter and fatter-tailed distributions. The function `two_norms` samples from such a fat-tailed distribution much like the one in the notebook `1-mildly-fat-tails`. In summary, it takes as a parameter $a$ where the larger the value the fatter the tails. For fat tails, MAD is considerably more efficient.

# In[4]:


"""
Efficiency as we occasionally sample from a higher STD normal distributon.
"""
a = [ _ for _ in range(0,21) ]

p = 0.01

efficiencies = [get_efficiency_sample(a_, p) for a_ in a]

plt.plot(a, efficiencies)
plt.xlabel('a')
plt.ylabel("Efficiency STD/MAD")
plt.xlim(0, 20)

