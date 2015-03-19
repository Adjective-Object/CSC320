from PIL import Image
import os
import os.path as path
import numpy as np
from scipy.misc import imread, imshow
from pylab import cm
import matplotlib.pyplot as plt
import matplotlib
import itertools, operator, random
from p3 import *

# set this to the directory that includes each of the actors
ACTORS_DIR = "processed_3"

# constants for images
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT


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
    filenames = filenames[0:min(100, len(filenames))]

    data_matrix = np.zeros((len(filenames), IMAGE_SIZE))

    for i, filename in list(enumerate(filenames)):
        try:
            path_to_file = path.join(data_dirname, filename)
            data = imread(path_to_file, True)
            flattened_data = np.ravel(data)
            
            data_matrix[i, :] = flattened_data / 255
        except Exception:
            debug("error loading file %s"%(path))

    return data_matrix

  

def extend_str(s, l):
    return s + " " * (l - len(s))

def pretty_table(table, num_format="{:.2f}"):
    table = [([" "+(item 
                if isinstance(item, str) 
                else (num_format.format(item) 
                        if isinstance(item, float) 
                        else str(item))) +" " 
                for item in row]) if row else row
            for row in table]

    width = len(table[0])
    row_widths = [max([len(row[i]) for row in table if row]) for i in range(width)]
       
    
    hbar = "-" * sum(row_widths) + "-" * (len(row_widths) -1)

    def make_tabular(t):    
        return ["|".join(
                map(lambda tup: 
                        extend_str(tup[1], row_widths[tup[0]]), 
                        enumerate(row))
                if row else [hbar]
                ) for row in t]
    
    
    return (" " + hbar + " " + "\n|" +
            "|\n|".join(
                make_tabular(table)
            ) + "|\n" +" " + hbar + " ")
    
def most_common(lst):
    return max(set(lst), key=lst.count) if len(lst) > 0 else None



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
        M = np.dot(X,X.T) # covariance matrix
        e,EV = np.linalg.eigh(M) # eigenvalues and eigenvectors
        tmp = np.dot(X.T,EV).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        S = np.sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - SVD used
        U,S,V = np.linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data

    # return the projection matrix, the variance and the mean
    return V,S,mean_X



def unflatten_face(flattened_face):
    ''' reshape a (1024) vector into a (32,32) image
    '''
    return np.reshape(flattened_face, (flattened_face.shape[0] / IMAGE_WIDTH, IMAGE_WIDTH))

def showall(imgs):
    ''' display a list of images as a grid 
        (takes either a numpy array of images or a list of numpy arrays that 
        are images)
    '''
    width = math.floor(math.sqrt(imgs.shape[0]))
    height = math.ceil(imgs.shape[0] * 1.0 / width)
    plt.figure()
    for i, img in enumerate(imgs):
        axes = plt.subplot(width, height, i+1)
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        imgplt = plt.imshow(img, cmap=cm.Greys_r)
        imgplt.set_interpolation('nearest')
        plt.title(str(i+1))
    plt.show()

def showall_flattened(imgs_flat):
    ''' reshape a set of (1024) vectors into (32,32) images, and display all of
        them in a greyscale grid
    '''
    print(imgs_flat)
    s=imgs_flat.shape
    unflattened = np.reshape(imgs_flat, (s[0],32,32))
    debug(unflattened.shape)
    showall(unflattened)
    
def show_flattened_face(flattened_face):
    ''' reshape a (1024) vector into a (32,32) image and display in greyscale
    '''
    unflattened = unflatten_face(flattened_face)
    plt.figure()
    imgplt = plt.imshow(unflattened, cmap=cm.Greys_r)
    imgplt.set_interpolation('nearest')
    plt.show()

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
    ''' TODO docs
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
            the number of bases from bases to consider when creating the
            subspace
    
        returns:
            the indecies of the projection in 'projections' that is closest to
            test_face projected onto projections.
    '''

    projected_training_faces = np.array(list(map (
        lambda training_face: project_to_space(training_face, bases[:k]),
        training_faces
    )))
    
    projected_test_faces = np.array(list(map (
        lambda test_face: project_to_space(test_face, bases[:k]),
        test_faces
    )))
    
    debug(projected_test_faces)
    debug(bases[:k])    
    
    reconstructed_test_faces = np.dot(projected_test_faces, bases[:k])
    
    showall_flattened(reconstructed_test_faces)
    
    # plt.figure()
    # plt.plot(projected_training_faces[:,0], projected_training_faces[:,1], 'ro')
    # plt.plot(projected_test_faces[:,0], projected_test_faces[:,1], 'go')
    # plt.show()
    
    return [closest_face(face, bases[:k], projected_training_faces)
                for face in test_faces]
      
# the "main function" for testing
def do_test():
    font = {'family' : 'normal',
            'size'   : 8}    
    matplotlib.rc('font', **font)
    
        # all the actor directories that we have
    actor_dirnames = [
                      "Adam_Sandler"
                     ,"Andrea_Anders" 
                     ,"Dianna_Agron"
                     ,"Ashley_Benson"
                     ,"Adrien_Brody"
                     ,"Christina_Applegate"
                     ,"Gillian_Anderson"
                     ,"Aaron_Eckhart"
                      ]
    random.shuffle(actor_dirnames)
    debug("actors: %s"%(", ".join(actor_dirnames)))

    # tools for mapping from data index to actor name
    data = [load_data(actor, "training") for actor in actor_dirnames]
    datalens = list(map(len, data))
    all_actor_faces = np.concatenate(data, axis=0)

    # debug(data[0])
    
    def get_name_from_ind(ind):
        dataind = 0
        while ind > datalens[dataind]:
            ind -= datalens[dataind]
            dataind += 1
        return actor_dirnames[dataind]
    
    debug("performing pca on all actors")
    components, single_values, avg_face = pca(all_actor_faces)
    # showall_flattened(np.concatenate(([avg_face], components[0:24]), axis=0))
    
    emptys = np.zeros((32,32))
    fulls =  np.empty((32,32))
    fulls.fill(255)
    max_dist = ssd(emptys, fulls)    
    
    
    results_table = [["k", "Name", "Accuracy"]]
    similarity_table = [["k", "actual", "percieved", "count"]]
    
    
    complete_num = 0
    complete_num_correct = 0
    for k in [100]:
        results_table.append(None)
        similarity_table.append(None)
        total_num_correct, total_num = 0, 0
        # results_table.append([k])
        for name in actor_dirnames:
            valid_faces = load_data(name, "validation")
            mean_faces = np.repeat(
                            np.array([avg_face]), 
                            valid_faces.shape[0],
                            axis=0)
            #valid_faces = np.subtract(valid_faces, mean_faces)
            #debug("vald:", valid_faces.shape)
            #debug("mean:", mean_faces.shape)
            #debug("diff:", valid_faces.shape)
            
            total_num += len(valid_faces)
            misses = []
            
            results = closest_projections(
                valid_faces,
                all_actor_faces,
                components,
                k)
                
            num_correct=0
            for index, distance in results:    
                out_name = get_name_from_ind(index)
                if name == out_name:
                    num_correct += 1
                else:
                    misses.append(out_name)
                '''
                debug(
                    out_name,
                    1-(distance / max_dist))
                '''
            # debug("%s :"%(extend_str(name,20)), num_correct/len(valid_faces))
            total_num_correct += num_correct
            results_table.append([k, name, num_correct/len(valid_faces)])
            mc =  most_common(misses)
            similarity_table.append([k, name, mc, misses.count(mc)])
       
        results_table.append(["", "(avg)", total_num_correct * 1.0 / total_num])
        complete_num += total_num
        complete_num_correct += total_num_correct 

    results_table.append(None)
    results_table.append(["", "(avg)", complete_num_correct * 1.0 / complete_num])
    
    debug(pretty_table(results_table))
    debug(pretty_table(similarity_table))
    

if __name__ == "__main__":
    do_test()
