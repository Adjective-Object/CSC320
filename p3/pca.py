import Image
import os
import os.path as path
import numpy as np
from scipy.misc import imread, imshow

# set this to the directory that includes each of the actors
ACTORS_DIR = "processed_3"

# constants for images
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT

# all the actor directories that we have
actor_dirnames = os.listdir(ACTORS_DIR)

def pca(X):
    """    Principal Component Analysis
        input: X, matrix with training data stored as flattened arrays in rows
        return: projection matrix (with important dimensions first), variance and mean.
        From: Jan Erik Solem, Programming Computer Vision with Python
        #http://programmingcomputervision.com/
    """

    # get dimensions
    num_data,dim = X.shape

    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim>num_data:
        # PCA - compact trick used
        M = dot(X,X.T) # covariance matrix
        e,EV = linalg.eigh(M) # eigenvalues and eigenvectors
        tmp = dot(X.T,EV).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        S = sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - SVD used
        U,S,V = linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data

    # return the projection matrix, the variance and the mean
    return V,S,mean_X

def load_data(actor, category):
    """
    Loads all the data for a given actor under the given category.
    All images within the directory must be the same size.

    :param actor: The actor directory name.
    :param category: The type of data(one of training, test, validation)
    :return: A NxM matrix where N is the number of images in the directory,
    and M is the size of each image, where each row is the flattened data
    of the image.
    """

    data_dirname = path.join(ACTORS_DIR, actor, category)
    filenames = os.listdir(data_dirname)

    data_matrix = np.zeros((len(filenames), IMAGE_SIZE))

    for i, filename in enumerate(filenames):
        path_to_file = path.join(data_dirname, filename)
        data = imread(path_to_file, True)
        flattened_data = np.ravel(data)

        data_matrix[i, :] = flattened_data

    return data_matrix

def get_average_face(faces):
    return np.mean(faces, axis=0)

def unflatten_face(flattened_face):
    return np.reshape(flattened_face, (flattened_face.shape[0] / IMAGE_WIDTH, IMAGE_WIDTH))

def show_flattened_face(flattened_face):
    unflattened = unflatten_face(flattened_face)
    imshow(unflattened)

data = load_data(actor_dirnames[0], "training")
avg_face = get_average_face(data)
show_flattened_face(avg_face)
