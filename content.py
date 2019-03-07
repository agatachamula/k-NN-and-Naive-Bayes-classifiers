from __future__ import division
import numpy as np
from scipy.spatial import distance



def hamming_distance(X, X_train):
    """
    :param X: set of objects that are going to be compared N1xD
    :param X_train: set of objects compared against param X N2xD
    Functions calculates Hamming distances between all objects from set X  and all object from set X_train. Resulting distances are returned as matrices.
    :return: Matrix stroing distances between objects X and X_train (N1xN2)
    """

    # N1 = X.shape[0]
    # D = X.shape[1]
    # N2 = X_train.shape[0]
    # result= np.zeros((N1,N2))

    # for k in range(0,N2):
    #    sum=0
    #    for i in range(0, N1):
    #        for j in range(0,D):
    #            if(X_train[k,j]!=X[i,j]):
    #                sum=sum+1
    #        result[i,k]=sum
    #         sum=0


    X = X.toarray()
    X_train = X_train.toarray()
    D=X.shape[1]
    return distance.cdist(X, X_train, metric='hamming') * D



def sort_train_labels_knn(Dist, y):
    """
    Function sorts labels of training data y accordingly to probabilities stored in matrix Dist.
    Function returns matrix N1xN2. In each row there are sorted data labels y accordingly to corresponding row of matrix Dist.
    :param Dist: Distance matrix between objects X and X_train N1xN2
    :param y: N2-element vector of labels
    :return: Matrix of sorted class labels ( use mergesort algorithm)
    """

    N1=Dist.shape[0]
    N2=Dist.shape[1]

    result=(np.argsort(Dist, axis=1, kind='mergesort'))

    for i in range(0,N1):
        for j in range(0,N2):
            result[i,j]=y[result[i,j]]


    return result

def p_y_x_knn(y, k):
    """
    Function calculates conditional probability p(y|x) for
    all classes and all objects from test set using KNN classifier
    :param y: matrix of sorted labels for training set N1xN2
    :param k: number of nearest neighbours
    :return: matrix of probabilities for objects X
    """

    #print(k)
    #print(y[0,:])

    N1=y.shape[0]
    N2=y.shape[1]

    result=np.zeros((N1,4))


    for i in range(0,N1):
        class4=0
        class1=0
        class2=0
        class3=0
        for j in range(0,k):
            if(y[i,j]==4):
                class4=class4+1
            elif(y[i,j]==1):
                class1=class1+1
            elif(y[i,j]==2):
                class2=class2+1
            elif(y[i,j]==3):
                class3=class3+1
        result[i,0]=(class1/k)
        result[i,1]=class2/k
        result[i,2] = class3 / k
        result[i,3] = class4 / k

    return result


def classification_error(p_y_x, y_true):
    """
    Function calculates classification error
    :param p_y_x: matrix of predicted probabilities
    :param y_true: set of ground truth labels 1xN.
    Each row of matrix represents distribution p(y|x)
    :return: classification error
    """
    #print(p_y_x[0,:])
    #print(y_true)
    N1=p_y_x.shape[0]
    N2=p_y_x.shape[1]

    sum=0
    row = np.argsort(p_y_x, axis=1, kind='mergesort')

    for i in range(0,N1):
        if(row[i,-1]+1!=y_true[i]):
            sum=sum+1

    sum=sum/N1

    return sum



def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: validation data N1xD
    :param Xtrain: training data N2xD
    :param yval: class labels for validation data 1xN1
    :param ytrain: class labels for training data 1xN2
    :param k_values: values of parameter k that are going to be evaluated
    :return: function makes model selection with knn and results tuple best_error,best_k,errors), where best_error is the lowest
    error, best_k - value of k parameter that corresponds to the lowest error, errors - list of error values for
    subsequent values of k (elements of k_values)
    """
    sizeK=len(k_values)


    errors=np.zeros((1,sizeK))
    Dist = hamming_distance(Xval, Xtrain)
    yy = sort_train_labels_knn(Dist, ytrain)

    for i in range(0,sizeK):
        p_y_x=p_y_x_knn(yy,k_values[i])
        errors[0,i]=(classification_error(p_y_x,yval))

    sortedErrors = np.argsort(errors[0,:], axis=0, kind='mergesort')
    bestindex=sortedErrors[0]
    bestError=errors[0,bestindex]
    bestK=k_values[bestindex]



    return(bestError,bestK,errors[0])

def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: labels for training data 1xN
    :return: function calculates distribution a priori p(y) and returns p_y - vector of a priori probabilities 1xM
    """
    N=len(ytrain)
    p_y=np.zeros((1,4))

    class1=0
    class2=0
    class3=0
    class4=0
    for i in range(0,N):
        if (ytrain[i] == 4):
            class4 = class4 + 1
        elif (ytrain[i] == 1):
            class1 = class1 + 1
        elif (ytrain[i] == 2):
            class2 = class2 + 1
        elif (ytrain[i] == 3):
            class3 = class3 + 1

    p_y[0,0]=class1/N
    p_y[0, 1] = class2 / N
    p_y[0, 2] = class3 / N
    p_y[0, 3] = class4 / N

    return p_y




def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: training data NxD
    :param ytrain: class labels for training data 1xN
    :param a: parameter a of Beta distribution
    :param b: parameter b of Beta distribution
    :return: Function calculated probality p(x|y) assuming that x takes binary values and elements
    x are independent from each other. Function returns matrix p_x_y that has size MxD.
    """
    Xtrain = Xtrain.toarray()
    N = len(ytrain)
    D=Xtrain.shape[1]
    result = np.zeros((4, D))

    class1t = 0
    class2t = 0
    class3t = 0
    class4t = 0
    for i in range(0,N):
        if (ytrain[i] == 4):
            class4t = class4t + 1
        elif (ytrain[i] == 1):
            class1t = class1t + 1
        elif (ytrain[i] == 2):
            class2t = class2t + 1
        elif (ytrain[i] == 3):
            class3t = class3t + 1


    for i in range(0, D):
        class1 = 0
        class2 = 0
        class3 = 0
        class4 = 0
        for j in range(0,N):
            if (ytrain[j] == 4 and Xtrain[j,i]==1):
                class4 = class4 + 1
            elif (ytrain[j] == 1 and Xtrain[j,i]==1):
                class1 = class1 + 1
            elif (ytrain[j] == 2 and Xtrain[j,i]==1):
                class2 = class2 + 1
            elif (ytrain[j] == 3 and Xtrain[j,i]==1):
                class3 = class3 + 1
        result[0,i]=(class1+a-1)/(class1t+a+b-2)
        result[1, i] = (class2 + a - 1) / (class2t + a + b - 2)
        result[2, i] = (class3 + a - 1) / (class3t + a + b - 2)
        result[3, i] = (class4 + a - 1) / (class4t + a + b - 2)


    return result



def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: vector of a priori probabilities 1xM
    :param p_x_1_y: probability distribution p(x=1|y) - matrix MxD
    :param X: data for probability estimation, matrix NxD
    :return: function calculated probability distribution p(y|x) for each class with the use of Naive Bayes classifier.
     Function returns matrixx p_y_x of size NxM.
    """
    X = X.toarray()
    N=X.shape[0]
    D=X.shape[1]
    p_y_x = np.zeros((N, 4))

    for i in range(0,N):
        p_y_x[i, :] = np.prod(np.power(p_x_1_y, X[i, :]) * np.power((1 - p_x_1_y), (1 - X[i, :])), axis=1) * p_y
        p_y_x[i, :] /= np.sum(p_y_x[i, :])

    return p_y_x



def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: training setN2xD
    :param Xval: validation setN1xD
    :param ytrain: class labels for training data 1xN2
    :param yval: class labels for validation data 1xN1
    :param a_values: list of parameters a (Beta distribution)
    :param b_values: list of parameters b (Beta distribution)
    :return: Function makes a model selection for Naive Bayes - that is selects the best values of a i b parameters.
    Function returns tuple (error_best, best_a, best_b, errors) where best_error is the lowest error,
    best_a - a parameter that corresponds to the lowest error, best_b - b parameter that corresponds to the lowest error,
    errors - matrix of errors for each pair (a,b)
    """

    sizeA = len(a_values)
    sizeB = len(b_values)


    errors = np.zeros((sizeA, sizeB))


    py=estimate_a_priori_nb(ytrain)


    for i in range(0, sizeA):
        for j in range(0,sizeB):
            print("a=",a_values[i],"b=",b_values[j])
            p_x_y=estimate_p_x_y_nb(Xtrain,ytrain,a_values[i],b_values[j])
            p_y_x = p_y_x_nb(py[0], p_x_y, Xval)
            errors[i,j] = (classification_error(p_y_x, yval))


    bestError = np.min(errors)
    flatArrayPos=np.argmin(errors)
    indexB=flatArrayPos%sizeB
    indexA=flatArrayPos//sizeB

    bestA=a_values[indexA]
    bestB=b_values[indexB]

    return (bestError,bestA, bestB, errors)


#A=np.matrix([0,1,0,1,0])
#B=np.matrix([0,0,0,1,1])
#hamming_distance(A,B)