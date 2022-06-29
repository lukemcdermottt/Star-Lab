import time
from operator import mul
import numpy as np
import matplotlib.pyplot as plt

class MixtureOfGaussian():

  def __init__(self,X,K):
    #Initialize all parameters
    self.X = X
    self.D = self.X.shape[1]
    self.num_Clusters = K
    self.mu = runKMeans(K, X)
    self.prior = np.asarray([1/K]*K)
    self.cov = np.zeros((K,self.D,self.D))
    for i in range(self.num_Clusters):
      self.cov[i] = np.diag(np.arange(2304)+1)

  #Use old parameters to compute gamma, then use gamma to recompute parameters
  def train_EM(self, x, prior_old, mu_old, cov_old):
    for _ in range(1000):
      N = len(x) #N is number of patterns in data
      h = np.zeros((N,self.num_Clusters), dtype = np.float32) #h is gamma matrix, where gamma_ij = P(k=j | x_i)
      for cluster in range(self.num_Clusters):
        # Gamma_ik = Norm(X_i, mu_k, sigma_k^2) * (P(c=k) / P(X_i)
        h[:,cluster] = self.gaussian(x, mu_old[cluster], cov_old[cluster]) * prior_old[cluster]
      #Dividing the denominator
      h = np.divide(h, np.sum(h, axis = 1,keepdims = True))
      #Gamma is in shape NxK: Number of Patterns x Number of Clusters
      
      #Recalculate new parameters
      prior_new = np.sum(h,axis = 0) / N
      mu_new = np.divide(h.T @ x, np.sum(h, axis = 0, keepdims = True).T)
      #Find covariance
      cov_temp = np.zeros((self.num_Clusters,self.D))
      for cluster in range (self.num_Clusters):
        cov_temp[cluster] = np.divide(h[:,cluster].T @ ((x - mu_new[cluster])*((x-mu_new[cluster]))), np.sum(h[:,cluster]))
      cov_new = np.zeros((self.num_Clusters,self.D,self.D))
      for i in range(self.num_Clusters):
        cov_new[i] = np.diag(cov_temp[i])

      #Calculate change in parameters
      if np.linalg.norm(mu_new - mu_old) < 1e-3:
        break
      prior_old, mu_old, cov_old = prior_new, mu_new, cov_new

    #After training across M <= 1000 epochs, return new optimized parameters
    return prior_new, mu_new, cov_new, h

  
  #Computes Gaussian
  def gaussian(self, x, mu, cov):
    norm.ppf()
    d = np.shape(x)[1]
    mu = mu[None, :]
    first_term = 1/(np.sqrt(2)**d)
    print(first_term)
    second_term = 1/(np.sqrt(np.pi)**d)
    print(second_term)
    third_term = 1/(np.sqrt((np.linalg.det(cov))))
    print(third_term)
    dr = first_term * second_term * third_term
    #dr = 1/(np.sqrt((2* np.pi)**(d) * np.linalg.det(cov)))
    nr = (np.exp(-np.diag((x-mu)@(np.linalg.inv(cov))@((x-mu).T)/2)))
    return nr * dr

  

  #Returns parameters of MixtureOfGaussian
  def get_parameters(self):
    prior_1, mu_1, cov_1 = self.train_EM(self.X, self.prior, self.mu, self.cov)
    params = [prior_1, mu_1, cov_1]
    return params

def plotCurrent(X, Rnk, Kmus):
    N, D = X.shape
    K = Kmus.shape[0]

    InitColorMat = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1],
                             [0, 0, 0],
                             [1, 1, 0],
                             [1, 0, 1],
                             [0, 1, 1]])

    KColorMat = InitColorMat[0:K,:]

    colorVec = np.dot(Rnk, KColorMat)
    muColorVec = np.dot(np.eye(K), KColorMat)
    plt.scatter(X[:,0], X[:,1], c=colorVec)

    plt.scatter(Kmus[:,0], Kmus[:,1], s=200, c=muColorVec, marker='d')
    plt.axis('equal')
    #plt.show


def runKMeans(K,X):
    #print('Running K Means:')
    #load data file specified by fileString from Bishop book
    X = X

    #determine and store data set information
    N, D = X.shape

    #allocate space for the K mu vectors
    Kmus = np.zeros((K, D))

    #initialize cluster centers by randomly picking points from the data
    rand_inds = np.random.permutation(N)
    Kmus = X[rand_inds[0:K],:]

    #specify the maximum number of iterations to allow
    maxiters = 1000

    for iter in range(maxiters):
        #assign each data vector to closest mu vector as per Bishop (9.2)
        #do this by first calculating a squared distance matrix where the n,k entry
        #contains the squared distance from the nth data vector to the kth mu vector

        #sqDmat will be an N-by-K matrix with the n,k entry as specfied above
        sqDmat = calcSqDistances(X, Kmus)
        #given the matrix of squared distances, determine the closest cluster
        #center for each data vector

        #R is the "responsibility" matrix
        #R will be an N-by-K matrix of binary values whose n,k entry is set as
        #per Bishop (9.2)
        #Specifically, the n,k entry is 1 if point n is closest to cluster k,
        #and is 0 otherwise
        Rnk = determineRnk(sqDmat)
        KmusOld = Kmus
        plotCurrent(X, Rnk, Kmus)
        #print(Rnk)
        time.sleep(1)
        #recalculate mu values based on cluster assignments as per Bishop (9.4)
        Kmus = recalcMus(X, Rnk)
        #check to see if the cluster centers have converged.  If so, break.
        if np.sum(np.abs(KmusOld.reshape((-1, 1)) - Kmus.reshape((-1, 1)))) < 1e-6:
            #print(iter)
            break

    
    plotCurrent(X, Rnk, Kmus)
    return Kmus, Rnk

def calcSqDistances(X, Kmus):
  N, D = X.shape
  K = Kmus.shape[0]
  sqDmat = np.empty((N,K))
  for n in range(N):
    for k in range(K):
      sqDmat[n][k] = np.linalg.norm(X[n] - Kmus[k]) ** 2
  
  return sqDmat

def recalcMus(X, Rnk):
  return np.dot(X.T,Rnk).T / np.sum(Rnk, axis = 0)[:,None]


def determineRnk(sqDmat):
  N, K = sqDmat.shape
  for i in range(N):
    temp = np.zeros(K)
    temp[np.argmin(sqDmat[i])] = 1
    sqDmat[i] = temp

  return sqDmat
  
def eigsort(V, eigvals):
    # Sort the eigenvalues from largest to smallest. Store the sorted
    # eigenvalues in the column vector lambd.
    lohival = np.sort(eigvals)
    lohiindex = np.argsort(eigvals)
    lambd = np.flip(lohival)
    index = np.flip(lohiindex)
    Dsort = np.diag(lambd)
    
    # Sort eigenvectors to correspond to the ordered eigenvalues. Store sorted
    # eigenvectors as columns of the matrix vsort.
    M = np.size(lambd)
    Vsort = np.zeros((M, M))
    for i in range(M):
        Vsort[:,i] = V[:,index[i]]
    return Vsort, Dsort

def PCA(x, dim):
  n = x.shape[0]

  mean = np.mean(x, axis = 0, keepdims = True)  #Avg Pattern
  x = x - mean  #Zero mean the data

  cov = (x.T @ x) / n 
  e_values, e_vectors = np.linalg.eig(cov)  
  sort_vec, sort_val = eigsort(e_vectors, e_values) #sort vectors based on biggest eigenvalues

  #debugging, delete next two lines later
  #print(sort_val.diagonal()[:10])

  #plt.plot(range(len(sort_val)), sort_val.diagonal()) #Print Scree Plot
  #plt.show()

  PCA_coords = x @ sort_vec[:,:dim] #Apply transformaton

  """
  print('DIMENSIONS:')
  print('Cov shape:',cov.shape)
  print('E_val shape:',e_values.shape)
  print('E values:', sort_val)
  print('E_vec shape:',e_vectors.shape)
  print('sort_val shape:',sort_val.shape)
  print('sort_vec shape:',sort_vec.shape)
  print('mean shape:',mean.shape)
  print('x shape:',x.shape)
  print('PCA shape:', PCA_coords.shape)
  """
  return PCA_coords