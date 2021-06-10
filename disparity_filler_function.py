# Liscence Information

""" Disparity Filler Function model for filling disparity maps from edges
    by Benjamin Keltjens, Tom van Dijk and Guido de Croon 
"""

import tensorflow as tf
import numpy as np


def disparityFillerFunction(images, disparities, name='untextured_filler', **kwargs):
    # Generate filled disparity map with signifcant edges
    # Input:    images (tensor) - tensor of image
    #           disparities (tensor) - tensor of sparse disparities (batch, height, width, channels(1))
    # Return:   averaged_disps (tensor) - tensor of interpolated disparities filled up in the same shape as disparities
             
    def _getEuclideanNorm(x, y):
        # Return euclidian norm of two tensors
        # Input:    x (tensor)
        #           y (tensor)
        # Return: Euclidian norm (tensor)
        return tf.sqrt(x*x + y*y)


    def _getSobelEdgesN(image, N):
        # Given batch of images return tuple of sobel gradients in x and y for the batch for size N sobel filter
        # Input:    image (tensor) - tensor of image
        #           N (int) - Size of sobel filter
        # Return:   filtered_x (tensor) - sobel gradient in the x direction
        #           filtered_y (tensor) - sobel gradient in the y direction

        def createNumpySobel(N, axis):
            # Create sobel gradient filter in numpy
            # Based on: https://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size#10032882
            # Input:    N (int) - Size of sobel filter. Must be odd
            #           axis (int) - direction of gradient 0 - positive x and 1 - positive y
            # Return:   Kernel (numpy array) - Kernel for sobel gradient

            Kernel = np.zeros((N,N))
            p = [(j,i) for j in range(N) 
                for i in range(N) 
                if not (i == (N -1)/2. and j == (N -1)/2.)]

            for j, i in p:
                j_ = int(j - (N -1)/2.)
                i_ = int(i - (N -1)/2.)
                Kernel[j,i] = (i_ if axis==0 else j_)/float(i_*i_ + j_*j_)

            return Kernel
        
        Kernelx = tf.convert_to_tensor(createNumpySobel(N, 0), np.float32)
        Kernely = tf.convert_to_tensor(createNumpySobel(N, 1), np.float32)

        Kernelx = tf.reshape(Kernelx, [N, N, 1, 1])
        Kernely = tf.reshape(Kernely, [N, N, 1, 1])

        P = int(N/2)

        paddings = tf.constant([[0,0],[P, P], [P, P], [0, 0]]) # Create tensor describing padding
        padded_image = tf.pad(image, paddings, "SYMMETRIC") # Pad tensor with symmetric boundaries for better averaging

        filtered_x = tf.nn.conv2d(padded_image, Kernelx, strides=[1, 1, 1, 1], padding='VALID')
        filtered_y = tf.nn.conv2d(padded_image, Kernely, strides=[1, 1, 1, 1], padding='VALID')

        return filtered_x, filtered_y


    def _createSumKernel(N):
        # Create kernel to sum surrounding pixels
        # Input:  N (int) - Size of sum kernel
        # Return: Kernel (tensor) - Sum kernel of 1's shaped for input tensor

        np_array = np.ones((N,N)) # Create array of ones
        sumKernel = tf.convert_to_tensor(np_array, np.float32) # conver to numpy array

        return tf.reshape(sumKernel, [N, N, 1, 1]) # reshape array to apply on batches
    
    def _createAveragingKernel(N):
        # Create kernel to average surrounding pixels, not including current pixel
        # Input:  N (int) - Size of averaging kernel
        # Return: Kernel (tensor) - averaging kernel 
        kernel = np.ones((N,N))
    
        # Find distances of each pixel from centre and sum
        for i in range(N):
            for j in range(N):
                f_x = abs(i- int(N/2)) # Round Down
                f_y = abs(j - int(N/2))
                distance = np.sqrt(f_x**2 + f_y**2)
                if distance != 0: # If not centre pixel
                    kernel[i,j] = 1/distance   
        kernel_sum = kernel.sum() # find sum
        kernel /= kernel_sum # Divide each distance by sum of distances

        avgKernel = tf.convert_to_tensor(kernel, np.float32) # Convert kernel to tensor

        return tf.reshape(avgKernel, [N, N, 1, 1]) # reshape tensor to apply on batches

    def _convertBooltoFloat(_input):
        # Convert Boolean tensor to Float tensor 
        # Input:  _input (tensor) Input boolean tensor of arbitrary size
        # Return: (tensor) tensor of 1 and 0 floats from bools (tf.case works as well)

        shape = tf.shape(_input) # Get tensor shape
        ones = tf.ones(shape, dtype=tf.float32) # Create tensor of ones
        zeros = tf.zeros(shape, dtype=tf.float32) # Create tensor of zeros
               
        return tf.where(_input, ones, zeros) # Gather 1 and 0s based on bools

    def _getMask(_input):
        # Get boolean mask of active pixels (> 0.0)
        # Input:  _input (tensor) Tensor of floats of arbitrary size
        # Return: (tensor) tensor of 1 and 0s in float

        # return _convertBooltoFloat(tf.greater(_input, tf.constant(0.0, dtype=tf.float32))
        return tf.cast(tf.greater(_input, 0.0), dtype=tf.float32)

    def _getInvMask(_mask):
        # Get inverse of Mask
        # Input:  _mask (tensor) Tensor of floats (1.0 and 0.0)
        # Return: (tensor) Tensor of floats (1.0 and 0.0)

        return _convertBooltoFloat(tf.less(_mask, tf.constant(1.0, dtype=tf.float32)))
    
    def _getSparseDisparities(image, disparities):
        # Given image and disparity produce sparse map of disparities
        # Input:    image (tensor) - tensor of input image
        #           disparities (tensor) - map of disparity output from network
        # Return:   sparse_disparity (tensor) - map of sparse disparity

        N = 7 # Size of Sobel filter
        grey = tf.image.rgb_to_grayscale(image) # Convert to black and white
        sobel = _getSobelEdgesN(grey, N) # Calculate sobel edge tuples
        sobel_sqrt = _getEuclideanNorm(sobel[0],sobel[1]) # Calculate the euclidian norm for x and y sobel edges
        maxs = tf.reduce_max(sobel_sqrt, axis = [1,2,3]) # Find the max value of the edge gradients for each image in batch
        maxs_expanded = tf.expand_dims(tf.expand_dims(tf.expand_dims(maxs, axis = 1),2),3) # Expand max value dimensions for division
        sobel_normalised = tf.divide(sobel_sqrt, maxs_expanded) # Divide all normalised edge gradients by maximum to map to [0,1]
        sobel_filtered = tf.cast(tf.greater(sobel_normalised, 0.1), tf.float32) # Get mask of normalised sobel above threshold as float
        sparse_disparities = sobel_filtered*disparities # Multiply disparities by float mask to get anchor points (sparse disparities)

        return sparse_disparities

    def _propagation(sparse_disps, batch, channels, height, width):
        # Run propagation step of function to pull out sparse disparities to unfilled parts of input tensor
        # Inputs: sparse_disps (tensor) - tensor of sparse disparities (batch, height, width, channels(1))
        #         batch (int) - batch size
        #         channels (int) - number of channels
        #         height (int) - height of input tensor
        #         width (int) - width of input tensor
        # Return: filled_disps (tensor) - tensor of filled disparities (batch, height, width, channels(1))
        #         mask (tensor) - mask of active pixels in float values of 1 and 0s [not important, byproduct of using while function]
        #         inv_mask_initial (tensor) - initial bool tensor of inactive pixles float values of 1 and 0
        #         count (tensor) - int tensor of count variable


        sumKernel = _createSumKernel(3) # Create sum kernel
        mask = _getMask(sparse_disps) # Find mask of active pixels
        inv_mask_initial = _getInvMask(mask) # Get intial mask of inactive pixels
        
        count = tf.convert_to_tensor(0) # create counter as tensor

        def condition(sparse_disps, mask, count):
            # Condition function for while loop. Simply counts to iterations
            # Input:  sparse_disps (tensor) - tensor of sparse disparities (batch, height, width, channels(1))
            #         mask (tensor) - mask of active pixels in float values of 1 and 0s
            #         count (tensor) - int tensor of count variable
            # Return: cond (tensor) - conditional tesnor for while function

            return tf.less(tf.reduce_sum(mask), batch*channels*height*width)
            # cond = tf.less(count, iters) # Continue if less count less than specified iterations
            # return cond

        def body(sparse_disps, mask, count):
            # Body function of the while loop. Fills in adjacent inactive pixels of active pixels on each iteration
            # Input:  sparse_disps (tensor) - tensor of sparse disparities (batch, height, width, channels(1))
            #         mask (tensor) - mask of active pixels in float values of 1 and 0s
            #         count (tensor) - int tensor of count variable
            # Return: Same as input except updated

            mask = _getMask(sparse_disps) # Find current pixel mask
            inv_mask = _getInvMask(mask) # Get inverse of active mask -> inactive pixels
            sum_mask = tf.nn.conv2d(mask, sumKernel, strides = [1,1,1,1], padding="SAME") # Run sum over mask to find sum of active neighbouring pixels
            sum_img = tf.nn.conv2d(sparse_disps, sumKernel, strides = [1,1,1,1], padding="SAME") # Run sum over disparities to find sum of values of neightbouring pixels
            averaged = tf.divide(sum_img,sum_mask+0.00001) # Take average of surrounding pixels. Add 'epsilon' to avoid divide by zero
            masked_average = tf.multiply(averaged,inv_mask) # Multiply averaged by inv_mask to only keep average of previous inactive pixels
            sparse_disps = tf.add(sparse_disps, masked_average) # Add new active pixels to previous disparity map


            return sparse_disps, mask, count+1
            
        # Call while loop function
        filled_disps, mask, count = tf.while_loop(condition, body, [sparse_disps, mask, count])

        return filled_disps, inv_mask_initial, count#iters

    def _smoothing(filled_disps, inv_mask_initial, iters):
        # Run second pass of function to average values from propagation
        # Inputs: filled_disps (tensor) - tensor of filled disparities (batch, height, width, channels(1))
        #         inv_mask_initial (tensor) - first mask tensor of inactive pixels from first pass
        #         iters (int) - Number of iterations which average filter is passed
        # Return: averaged_disps (tensor) - tensor of averaged disparities (batch, height, width, channels(1))
        N = 5 # Kernel Size
        P = int(N/2) # Padding Size
        avgKernel = _createAveragingKernel(N) # Create averaging kernel
        mask = _getInvMask(inv_mask_initial) # Get inverse mask of this, activated pixels, at the same step from first pass

        count = 0  # create count variable counting backwards from 

        def condition(filled_disps, count):
            # Condition function for while loop. Simply returns false, to terminate loop, when reached number of iterations
            # Input:  filled_disps (tensor) - tensor of filled disparities (batch, height, width, channels(1))
            #         count (tensor) - int tensor of count variable counting down to 0
            # Return: cond (tensor) - conditional tensor for while function

            cond = tf.less(count, iters) # Check if count has reached iterations
            return cond

        def body(filled_disps, count):
            # Body function for while loop. Average over filled pixels
            # Essentially performs averaging on the pixels in the opposite order that they were generated in the first pass
            # Input:  filled_disps (tensor) - tensor of filled disparities (batch, height, width, channels(1))
            #         count (tensor) - int tensor of count variable counting down to 0
            # Return: filled_disps (tensor) - tensor of averaged disparities (batch, height, width, channels(1))
            #         count (tensor) - int tensor of count variable counting down to 0 [not useful]

            paddings = tf.constant([[0,0],[P, P], [P, P], [0, 0]]) # Create tensor describing padding
            padded_disps = tf.pad(filled_disps, paddings, "SYMMETRIC") # Pad tensor with symmetric boundaries for better averaging
            averaged = tf.nn.conv2d(padded_disps, avgKernel, strides = [1,1,1,1], padding="VALID") # Pass averaging filter over filled_disparity map
            filled_disps = tf.add(tf.multiply(filled_disps, mask), tf.multiply(averaged, inv_mask_initial)) # Add unmodified active pixels to modified deactivated pixels
            return filled_disps, count+1 # Return updated values

        averaged_disps, count = tf.while_loop(condition, body, [filled_disps, count])
        return averaged_disps



    with tf.variable_scope(name):
        # Get shape of input tensor
        
        # Get sparse disparity map
        sparse_disparities = _getSparseDisparities(images, disparities)  

        # Find shape of disparity map
        tensor_shape  = sparse_disparities.get_shape().as_list()
        _num_batch    = tensor_shape[0]
        _height       = tensor_shape[1]
        _width        = tensor_shape[2]
        _num_channels = tensor_shape[3]

        # Run propagation step on sparse disparities
        filled_disps, inv_mask_initial, iters = _propagation(sparse_disparities, _num_batch, _num_channels, _height, _width)
        
        # Run second pass to average out disparities
        averaged_disps = _smoothing(filled_disps, inv_mask_initial, iters)

        # Return averaged filled disparity
        return averaged_disps
