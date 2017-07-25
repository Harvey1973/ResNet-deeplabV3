import sys
import numpy as np
import pickle
import os
import cv2

data_dir = 'cifar10_data'
full_data_dir = 'cifar10_data/cifar-10-batches-py/data_batch_'
vali_dir = 'cifar10_data/cifar-10-batches-py/test_batch'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3
NUM_CLASS = 10

TRAIN_RANDOM_LABEL = False # Want to use random label for train data?
VALI_RANDOM_LABEL = False # Want to use random label for validation?

NUM_TRAIN_BATCH = 5 # How many batches of files you want to read in, from 0 to 5)
EPOCH_SIZE = 10000 * NUM_TRAIN_BATCH

def _read_one_batch(path):
    '''
    THe training data contains five data batched in total. Validation data has one
    batch , returns one batch of data and annotations 
    param path : data path
    return image numpy arrays and label arrays
    '''
    fo = open(path,'rb')
    dicts = pickle.load(fo,encoding="bytes")
    fo.close()
    print(dicts.keys())
    data = dicts[b'data']
    label = np.array(dicts[b'labels'])
    return data, label
def read_in_all_images(address_list, shuffle = True):
    '''
    reads all training and validation data ,and perform shuffling if needed
    returns the image and label arrays
    param address_list : a list of path for pickle files
    return : data in form of [num_images, Hieght, Width, Channel]  labels in [num_images]
    '''
    data = np.array([]).reshape(0,IMG_HEIGHT*IMG_HEIGHT*IMG_DEPTH)
    label = np.array([])

    for address in address_list :
        print('reading images from'+ str(address))
        batch_data , batch_label = _read_one_batch(address)
        data = np.concatenate((data,batch_data))
        label = np.concatenate((label,batch_label))
    num_data = len(label)
    data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
    data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))
    
    if shuffle is True:
        print('shuffling')
        order = np.random.permutation(num_data)
        data = data[order,...]
        label = label[order]
    data = data.astype(np.float32)
    return data, label
def horizontal_flip(image, axis):
    '''
    Flip an image at 50% possibility
    :param image: a 3 dimensional numpy array representing an image
    :param axis: 0 for vertical flip and 1 for horizontal flip
    :return: 3D image after flip
    '''
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = cv2.flip(image, axis)

    return image
def whitening_image(image_np):
    '''
    Performs per_image_whitening
    :param image_np: a 4D numpy array representing a batch of images
    :return: the image numpy array after whitened
    '''
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        std = np.max([np.std(image_np[i,...]), 1.0/np.sqrt(IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH)])
        image_np[i,...] = (image_np[i,...] - mean) / std
    return image_np


def random_crop_and_flip(batch_data, padding_size):
    '''
    Helper to random crop and random flip a batch of images
    :param padding_size: int. how many layers of 0 padding was added to each side
    :param batch_data: a 4D batch array
    :return: randomly cropped and flipped image
    '''
    cropped_batch = np.zeros(len(batch_data) * IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH).reshape(
        len(batch_data), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+IMG_HEIGHT,
                      y_offset:y_offset+IMG_WIDTH, :]

        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)

    return cropped_batch

def prepare_train_data(padding_size):
    '''
    Read all the train data into numpy array and add padding_size of 0 paddings on each side of the
    image
    :param padding_size: int. how many layers of zero pads to add on each side?
    :return: all the train data and corresponding labels
    '''
    path_list = []
    for i in range(1, NUM_TRAIN_BATCH+1):
        path_list.append(full_data_dir + str(i))
    data, label = read_in_all_images(path_list)
    
    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
    
    return data, label


def read_validation_data():
    '''
    Read in validation data. Whitening at the same time
    :return: Validation image data as 4D numpy array. Validation labels as 1D numpy array
    '''
    validation_array, validation_labels = read_in_all_images([vali_dir])
    validation_array = whitening_image(validation_array)

    return validation_array, validation_labels