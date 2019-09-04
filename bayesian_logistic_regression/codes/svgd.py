import numpy as np
from scipy.spatial.distance import pdist, squareform

class SVGD():
    def __init__(self, theta, lnprob, bandwidth = -1, stepsize = 1e-3, alpha = 0.9, optType = 'adag'):
        if theta is None or lnprob is None: raise ValueError('theta or lnprob cannot be None!')
        self.theta = theta; self.lnprob = lnprob; self.stepsize = stepsize; self.alpha = alpha; self.optType = optType;
        if np.ndim(bandwidth) == 0: self.bandwidth = [bandwidth]
        else: self.bandwidth = bandwidth
        self.M, self.D = theta.shape
        self.iter = -1
        self.fudge_factor = 1e-6; self.historical_grad = 0;
    
    def __svgd_kernel(self):
        theta = self.theta
        M = self.M; D = self.D
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist)**2
        Kxy = np.zeros((M,M))
        dxkxy = np.zeros((M,D))
        for h in self.bandwidth:
            if h < 0: # if h < 0, using median trick
                h = np.median(pairwise_dists)  
                h = 0.5 * h / np.log(M+1)

            # compute the rbf kernel
            local_Kxy = np.exp( -pairwise_dists / h / 2)
            Kxy += local_Kxy

            local_dxkxy = -np.matmul(local_Kxy, theta)
            sumkxy = np.sum(local_Kxy, axis=1)
            for i in range(D): local_dxkxy[:, i] += np.multiply(theta[:,i],sumkxy)
            dxkxy += local_dxkxy / h
        
        return (Kxy, dxkxy)
 
    def update(self, n_iter = 10, debug = False):
        theta = self.theta; alpha = self.alpha; optType = self.optType;
        for self.iter in range(self.iter+1, self.iter+n_iter+1):
            if debug and self.iter % 100 == 0: print 'iter ' + str(self.iter) 
            lnpgrad = self.lnprob(theta)
            # calculating the kernel matrix
            kxy, dxkxy = self.__svgd_kernel()  
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / theta.shape[0]  
                
            if optType == 'adag':
                if self.iter == 0:
                    self.historical_grad += grad_theta ** 2
                else:
                    self.historical_grad = alpha * self.historical_grad + (1 - alpha) * (grad_theta ** 2)
                        
                adj_grad = np.divide(grad_theta, self.fudge_factor+np.sqrt(self.historical_grad))
                theta += self.stepsize * adj_grad 
            elif optType == 'gd':
                theta += self.stepsize * grad_theta
            elif optType == 'sgd':
                theta += self.stepsize/pow(self.iter+1., alpha) * grad_theta
            else: raise ValueError('unknown \'optType\': \'%s\'!'%optType)
                
        return theta

