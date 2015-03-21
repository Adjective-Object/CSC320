import itertools, operator, random, math, os

from PIL import Image
from scipy.misc import imread, imshow, imsave
import matplotlib.pyplot as plt

from scipy.misc import imread
import numpy as np

from pretty_table import *
from debug_pca import *

# set this to the directory that includes each of the actors
ACTORS_DIR = "processed_3"
MISS_DIR = "misses"

# constants for images
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT

def load_data(actor, actors_dir, category):
    """
    Loads all the data for a given actor under the given category.
    All images within the directory must be the same size.

    :param actor: The actor directory name.
    :param category: The type of data(one of training, test, validation)
    :return: A NxM matrix where N is the number of images in the directory,
    and M is the size of each image, where each row is the flattened data
    of the image.
    """

    data_dirname = os.path.join(actors_dir, actor, category)
    filenames = os.listdir(data_dirname)
    filenames = filenames[0:min(100, len(filenames))]

    data_matrix = np.zeros((len(filenames), IMAGE_SIZE))

    for i, filename in list(enumerate(filenames)):
        try:
            path_to_file = os.path.join(data_dirname, filename)
            data = imread(path_to_file, True)
            flattened_data = np.ravel(data)

            data_matrix[i, :] = flattened_data / 255
        except Exception:
            debug("error loading file %s"%(os.path))

    return data_matrix

def most_common(lst):
    return max(set(lst), key=lst.count) if len(lst) > 0 else None

def unflatten_face(flattened_face):
    ''' reshape a (1024) vector into a (32,32) image
    '''
    return np.reshape(flattened_face, (flattened_face.shape[0] / IMAGE_WIDTH, IMAGE_WIDTH))



def pca(X):
    """ Principal Component Analysis
        input: X, matrix with training data stored as flattened arrays in rows
        return: projection matrix (with important dimensions first), variance
        and mean.
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
        M = np.dot(X,X.T) # covariance matrix
        e,EV = np.linalg.eigh(M) # eigenvalues and eigenvectors
        tmp = np.dot(X.T,EV).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want

        # invert the eigenvalues and corresponding eigenvectors of negatives
        for i, val in enumerate(e):
            if val < 0:
                EV[i] = -EV[i]
                e[i]  = -e[i]

        # reverse since eigenvalues are in increasing order
        S = np.sqrt(np.abs(e))[::-1]
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - SVD used
        U,S,V = np.linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data

    # return the projection matrix, the variance and the mean
    return V,S,mean_X

def ssd(img1, img2):
    ''' given two same-dimension images, returns the sum of squared differences
        (euclidian distance) between the images
    '''
    return np.sum((img1 - img2) ** 2 )


def project_to_space(vec, bases):
    return np.dot(bases, vec)

def distance_to_space(offset_face, components):
    ''' given a set of k basis vectors and a single vector
        (typically the face offset by the mean face for an actor)
        give the euclidian distance between the offset_face and the closest
        point in the subspace
    '''
    projection = project_to_space(offset_face, components)
    return ssd(projection, offset_face)

def closest_face(test_face, bases, projected_training_faces):
    ''' returns the closest face in `projected_training_faces` to test_face
        projected onto the subspace defined by bases
    '''
    projected_test_face = project_to_space(test_face, bases)

    def ssd_to_test(face):
        return ssd(projected_test_face, face)

    distances = map(ssd_to_test, projected_training_faces)

    index, min_distance = min(
        enumerate(distances),
        key=lambda tuple_index_distance: tuple_index_distance[1]
    )

    return index, min_distance

def project_and_reconstruct(test_faces, bases):
    ''' projects a set of faces onto a subspace with bases `bases`

        return:
            the the closes image in the subspace from the `bases`
    '''

    projected_test_faces = np.array(list(map (
        lambda test_face: project_to_space(test_face, bases),
        test_faces
    )))

    reconstructed_test_faces = np.dot(projected_test_faces, bases)

    return reconstructed_test_faces

def closest_projections(test_faces, training_faces, bases, k):
    ''' finds the closest face to test_face i

        test_faces:
            the faces that are being queried against the subspaces
            (offset by the mean face)
        training_faces:
            a list of training faces from which bases was constructed
        bases:
            the bases of the subspace
        k:
            the number of bases in `bases` to consider when creating the
            subspace

        returns:
            the indecies of the projections from `trainign_faces` that are
            closest to (test_face projected onto the subspace bases[:k]).
    '''

    projected_training_faces = np.array(list(map(
        lambda training_face: project_to_space(training_face, bases[:k]),
        training_faces
    )))

    # showall_flattened(project_and_reconstruct(test_faces, bases[:k]))

    # plt.figure().canvas.set_window_title("anaconda")
    # plt.plot(projected_training_faces[:,0], projected_training_faces[:,1], 'ro')
    # plt.plot(projected_test_faces[:,0], projected_test_faces[:,1], 'go')
    # plt.show()

    return [closest_face(face, bases[:k], projected_training_faces)
                for face in test_faces]

# the "main function" for testing
def do_test(actors_dir="processed_3", 
            judge_dir="validation",
            display_similarity_table=False,
            k_values=[2, 5, 10, 20, 50, 80, 100, 150, 200],
            SAVE_FACE_MISSES=False):

    # all the actor directories that we have, as well as the gender of the
    # actors (m/f only for runtime purposes)
    ismale_map = {
      "Adam_Sandler" : True
     ,"Andrea_Anders": False
     ,"Dianna_Agron" : False
     ,"Ashley_Benson": False
     ,"Adrien_Brody" : True
     ,"Christina_Applegate": False
     ,"Gillian_Anderson" : False
     ,"Aaron_Eckhart": True
    }

    if SAVE_FACE_MISSES and not os.path.exists(MISS_DIR):
        debug("making missed face directory")
        os.mkdir(MISS_DIR)

    actor_dirnames = ismale_map.keys()
    debug("actors: %s"%(", ".join(actor_dirnames)))

    # load actor from dirnames
    data = [load_data(actor, actors_dir, "training") for actor in actor_dirnames]
    all_actor_faces = np.concatenate(data, axis=0)

    # setup for mapping actor name back to index in all_actor_facs
    datalens = list(map(len, data))
    def get_name_from_ind(ind):
        dataind = 0
        while ind > datalens[dataind]:
            ind -= datalens[dataind]
            dataind += 1
        return actor_dirnames[dataind]

    debug("performing pca on all actors")
    components, single_values, avg_face = pca(all_actor_faces)

    emptys = np.zeros((32, 32))
    fulls = np.empty((32, 32))
    fulls.fill(255)
    max_dist = ssd(emptys, fulls)

    # list of results indexed face_comparisons[k][number] = (real, matched)
    face_comparisons = {}

    debug("matching faces back to bases")
    for k in k_values:
        debug("k =", k)
        face_comparisons[k] = {}
        for name in actor_dirnames:

            # check against the validation set
            valid_faces = load_data(name, judge_dir)[0:10]

            results = closest_projections(
                valid_faces,
                all_actor_faces,
                components,
                k
            )

            # put the result in the comparisons dict
            face_comparisons[k][name] = [
                get_name_from_ind(index)
                    for index, distance in results
            ]

            if SAVE_FACE_MISSES:
                for i, (index, distance) in enumerate(results):
                    mnam = get_name_from_ind(index)
                    if mnam != name:
                        debug("saving missed faces %s %s"%(name, mnam))
                        realface = unflatten_face(
                                project_and_reconstruct(
                                    valid_faces[i:i+1],
                                    components[:k]) [0])

                        missface = unflatten_face(
                                project_and_reconstruct(
                                    all_actor_faces[index:index+1],
                                    components[:k]) [0])

                        joinface = np.empty((32, 64))
                        joinface[:,:32] = realface
                        joinface[:,32:]  = missface

                        imsave(
                            MISS_DIR+"/k%s_%s_%s_%s.png"%(k, name, i, mnam),
                            joinface
                        )

    debug("building tables...")

    print(pretty_table(build_results_table(face_comparisons, ismale_map)))

    if (display_similarity_table):
        similarity_table = build_similarity_table(face_comparisons)
        print(pretty_table(similarity_table))


if __name__ == "__main__":
    do_test()
