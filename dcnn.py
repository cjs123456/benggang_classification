import os
import inspect
import numpy as np
num_hops = 3  # number of diffusion convolution hop
num_features = 75  # number of diffusion convolution feature
def F2ADJ(MAP,X,h,w):
    '''
    Construct probability transition matrix
    Args:
        MAP: distance between nodes
        X: feature matrix
        h: height
        w: wideth
    Returns:
        A: probability transition matrix of each node
    '''
    Neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]#8
    A=np.zeros([h*w,h*w])
    for i in range(0,h):
        for j in range(0,w):
            for n in range(0,8):
                dr,dc=Neighbors[n]
                i2=i+dr
                j2=j+dc
                if i2<0 or i2>=h or j2<0 or j2>=w:
                    continue

                A[i*w+j,i2*w+j2]=1/(abs(MAP[i*w+j,n])+abs(MAP[i2*w+j2,n-4]))

                if X[i*w+j,0]-X[i2*w+j2,0]<0:
                    A[i * w + j, i2 * w + j2]=A[i*w+j,i2*w+j2]*(-1)

                #if MAP[i*w+j,n]<0:
                #    A[i * w + j, i2 * w + j2]=A[i*w+j,i2*w+j2]*(-1)
            #A[i*w+j,:]=normalization(A[i*w+j,:])
    return A

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def run_node_classification3(): # integrates multiple transfer matrixs of different data areas
    current_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    X_list, Y_list, A_list = locals(), locals(), locals()
    numlist = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
    # numlist = [1, 2, 3, 4, 5]
    for num in numlist:
        # path = "%s/data/DEM/" + str(i) + "/" % (current_dir,)
        path = current_dir + "/data/DEM/" + str(num) + "/"
        XX = np.loadtxt(path + 'feature.txt', delimiter=',', dtype=np.float32)
        AA = np.loadtxt(path + 'AA.txt', delimiter=',', dtype=np.float32)
        X = np.zeros((len(XX), len(XX[0]) + len(AA[0])))
        X[:, :len(XX[0])] = XX[:, :]
        X[:, len(XX[0]):] = AA

        MAP = np.loadtxt(path + 'MAP.txt', delimiter=',', dtype=np.float32)
        A = F2ADJ(MAP, XX, 18, 24)
        X_list['X_' + str(num)] = X
        A_list['A_' + str(num)] = A

    X_1, X_2, X_3, X_4, X_5,X_6,X_7,X_8,X_9,X_10 = X_list['X_1'], X_list['X_2'], X_list['X_3'], \
        X_list['X_4'], X_list['X_5'],X_list['X_11'],X_list['X_12'],X_list['X_13'],X_list['X_14'],X_list['X_15']

    A_1, A_2, A_3, A_4, A_5, A_6, A_7,A_8,A_9,A_10 = A_list['A_1'], A_list['A_2'], A_list['A_3'], \
        A_list['A_4'], A_list['A_5'],A_list['A_11'],A_list['A_12'],A_list['A_13'],A_list['A_14'],A_list['A_15']

    FINAL_A = np.zeros((18 * 24 * 10, 18 * 24 * 10)) #要改！

    FINAL_X0 = np.vstack((X_1, X_2))
    FINAL_X0 = np.vstack((FINAL_X0, X_3))
    FINAL_X0 = np.vstack((FINAL_X0, X_4))
    FINAL_X0 = np.vstack((FINAL_X0, X_5))
    FINAL_X0 = np.vstack((FINAL_X0, X_6))
    FINAL_X0 = np.vstack((FINAL_X0, X_7))
    FINAL_X0 = np.vstack((FINAL_X0, X_8))
    FINAL_X0 = np.vstack((FINAL_X0, X_9))
    FINAL_X0 = np.vstack((FINAL_X0, X_10))

    L = len(A_1)
    FINAL_A[:L, :L] = A_1
    FINAL_A[L:2 * L, L:2 * L] = A_2
    FINAL_A[2 * L:3 * L, 2 * L:3 * L] = A_3
    FINAL_A[3 * L:4 * L, 3 * L:4 * L] = A_4
    FINAL_A[4 * L:5 * L, 4 * L:5 * L] = A_5
    FINAL_A[5 * L:6 * L, 5 * L:6 * L] = A_6
    FINAL_A[6 * L:7 * L, 6 * L:7 * L] = A_7
    FINAL_A[7 * L:8 * L, 7 * L:8 * L] = A_8
    FINAL_A[8 * L:9 * L, 8 * L:9 * L] = A_9
    FINAL_A[9 * L:10 * L, 9 * L:10 * L] = A_10

    if num_features == 23:
        FINAL_X = np.hstack((FINAL_X0[:, :7], FINAL_X0[:, -16:]))
    if num_features == 75:
        FINAL_X = FINAL_X0
    if num_features == 52:
        FINAL_X = FINAL_X0[:, 7:-16]

    for i in range(len(FINAL_X[0])):
        FINAL_X[:, i] = normalization(FINAL_X[:, i])

    num_nodes = FINAL_A.shape[0]

    return FINAL_A, FINAL_X

def A_to_diffusion_kernel(A, k):
    """
    Computes [A**0, A**1, ..., A**k]

    :param A: 2d numpy array
    :param k: integer, degree of series
    :return: 3d numpy array [A**0, A**1, ..., A**k]
    """
    assert k >= 0

    Apow = [np.identity(A.shape[0], dtype=np.int)]

    if k > 0:
        d = A.sum(0)
        for m in range(0, len(A)):
            A[:,m] = A[:,m]/(d[m]+1.0)
        Apow.append(A)

        for i in range(2, k + 1):
            Apow.append(np.dot(A / (d + 1.0), Apow[-1]))

        a = np.asarray(Apow, dtype='float32')
    else:
        a = np.asarray(Apow, dtype='float32')

    return  np.transpose(a, (1, 0, 2))
