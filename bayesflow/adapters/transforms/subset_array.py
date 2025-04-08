import numpy as np
from keras.saving import register_keras_serializable as serializable

from .elementwise_transform import ElementwiseTransform



@serializable(package="bayesflow.adapters")
class SubsetArray(ElementwiseTransform):
    """
    A transform to reduce the dimensionality of arrays output by the summary network
    Sometimes, the simulators may return larger arrays than we want to use in our networks, so it would be great to have some subsetting adapter transforms.

    We need:

    Subsetting within an axis (taking only some elements) 
    while keeping the number of axes the same. This is essentially 
    the np.take functionality so we might want to call this transform
    take. 
    
    In contrast to np.take, I would make axis a mandatory 
    arguments or default it to the last axis. 
    Example: adapter.take("x", 1:3, axis = -1)

    Subsetting using a random set of indices (of user-specified size) 
    within an axis. We might call this subsample. Internally it would 
    call take after sampling the indices. 
    Example: adapter.subsample("x", size = 3, axis = -1)
    Removing an axis of length one. Following numpy, 
    I would call this transform squeeze: Example: adapter.squeeze("x", axis = 1)

    """

    def __init__(self, forward = str, inverse = str ):
        
        super().__init__()
    




    def take(self, data, indices, axis = -1): 
        # ithink that indices needs to be a list or a slice, if its a list then 
        # we can have that subsample provides a list of random slices 
        # warn if no axis is provided 
        # my question is how does np.take deal with a list of numbers? 
            # it will gladly take a listof numbers even non consecutive ones , 
            # it also warns for out of bounds 
        return np.take(data, indices, axis)
    
    
    def subsample(self, data, sample_size, axis): 
        
        max_sample_size = data.shape[axis]

        sample_indices  = np.random.permutation(max_sample_size)[0:sample_size-1] # random sample without replacement 

        self.take(data, sample_indices, axis)


       

