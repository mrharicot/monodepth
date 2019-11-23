# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *
from scipy import misc

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--encoder',          type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--image_path',       type=str,   help='path to the image', required=True)
parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', required=True)
parser.add_argument('--input_height',     type=int,   help='input height', default=256)
parser.add_argument('--input_width',      type=int,   help='input width', default=512)

args = parser.parse_args()

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def test_simple(params):
    """Test function."""

    left  = tf.placeholder(tf.float32, [1, args.input_height, args.input_width, 3])
    right  = tf.placeholder(tf.int32, [1, args.input_height, args.input_width, 1])
    model = MonodepthModel(params, "test", left, None)

    input_image = scipy.misc.imread(args.image_path, mode="RGB")
    label_image = scipy.misc.imread("/home/krishna/Pictures/bochum_000000_024196_gtFine_labelIds.png")
    #cv2.imshow("Frag", input_image)
    #cv2.waitKey(0)
    
    
    original_height, original_width, num_channels = input_image.shape
    input_image = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')
    label_image = scipy.misc.imresize(label_image, [args.input_height, args.input_width])
   
    #label_image = label_image.astype(np.float32)
    
    input_image = input_image.astype(np.float32) / 255
    #input_images = np.stack((input_image, np.fliplr(input_image)), 0)
    input_image = tf.reshape(input_image, [1, args.input_height, args.input_width, 3])
    label_image = tf.reshape(label_image, [1, args.input_height, args.input_width, 1])
    
    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    input_image = sess.run(input_image)
    #label_image = sess.run(label_image)
    #print(label_image[0, int(args.input_height/2), int(args.input_width/2),0])
    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
    
    '''
    all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    for i in all_variables_list:
        print(i)
    '''
    
    # RESTORE
    restore_path = args.checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)

    #disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
    #disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)
    disp_pp = sess.run(model.logit, feed_dict={left: input_image})
    
    #lab = sess.run(model.right, feed_dict={right: label_image})
    #print()
    #print(lab[0, int(args.input_height/2), int(args.input_width/2),0])
    #loss, onehot, inp, chk = sess.run([model.total_loss, model.one_hot_label, model.logit, model.right], feed_dict={left: input_image, right: label_image})
    #print(loss)
    #print(onehot[1211][:])
    #print(inp[0, 0, 0, 0])
    #chk = chk.flatten()
    #print(chk[1211])
    output_directory = os.path.dirname(args.image_path)
    output_name = os.path.splitext(os.path.basename(args.image_path))[0]

    np.save(os.path.join(output_directory, "{}_disp.npy".format(output_name)), disp_pp)
    copy_image = np.zeros((256,512,1), np.uint8)
    #img = cv2.imread("/home/krishna/magic.png")
    
    for i in range(0, 256):
        for j in range(0, 512):
            b = np.argmax(disp_pp[0,i,j,:])
            #print(disp_pp[0,i,j,:])
            copy_image[i][j] = b*5
            #print(b)
            #cv2.imshow("Frag", img)
            #cv2.waitKey(0)
            #break
        #break
    cv2.imwrite("/home/krishna/magic1.png",copy_image)
    for i in range(0, 33):
        disp_to_img = scipy.misc.imresize(disp_pp[0,:,:,i].squeeze(), [original_height, original_width])
        plt.imsave(os.path.join(output_directory, "{}_disp.png".format(output_name + str(i+50))), disp_to_img, cmap='plasma')
    
    print('done!')

def main(_):

    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        full_summary=False)

    test_simple(params)

if __name__ == '__main__':
    tf.app.run()
