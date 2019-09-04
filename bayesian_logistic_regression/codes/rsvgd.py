import numpy as np
from numpy import newaxis

class RSVGD:
    def __init__(self, theta, updInfo, bandwidth=-1, stepsize=1e-3, alpha=0.9, optType='gd'):
        if theta is None or updInfo is None: raise ValueError('theta or updInfo cannot be None!')
        self.theta = theta; self.updInfo = updInfo; self.stepsize = stepsize; self.alpha = alpha; self.optType = optType;
        if np.ndim(bandwidth) == 0: self.bandwidth = [bandwidth]
        else: self.bandwidth = bandwidth
        self.M, self.D = theta.shape
        self.iter = -1
        self.fudge_factor = 1e-6; self.historical_grad = 0;
    
    def update(self, n_iter=10, debug=False):
        theta = self.theta; alpha = self.alpha; optType = self.optType;
        M = self.M; D = self.D;

        for self.iter in range(self.iter+1, self.iter+n_iter+1):
            if debug and self.iter % 100 == 0: print 'iter ' + str(self.iter) 
            # M*D, M*D*D, M*D,    M*D
            gradp, Ginv, gradDet, gradGinv = self.updInfo(theta)
            totScore = np.matmul((gradp+gradDet/2.)[:,newaxis,:], Ginv).reshape(M,D) + gradGinv # "matmul" cannot be replaced with "dot" here! totScore: M*D
            Dxy = theta[:,newaxis,:] - theta[newaxis,:,:] # M*M*D, expectation over the first index
            sqDxy = (Dxy**2).sum(axis=2) # M*M
            DxyGinv = np.matmul(Dxy, Ginv)

            vect = np.zeros((M,D))
            for h in self.bandwidth:
                if h < 0: # if h < 0, using median trick
                    h = np.median(sqDxy)  
                    h = 0.5 * h / np.log(M+1)

                # compute the rbf kernel
                Kxy = np.exp( -sqDxy / h / 2)
                vect += (((( -DxyGinv/h + totScore[:,newaxis,:] )*Dxy).sum(axis=2)*Kxy /h**2).T[:,:,newaxis] * DxyGinv).sum(axis=1)\
                    + np.matmul((Kxy/h).dot(totScore)[:,newaxis,:], Ginv).reshape(M,D)\
                    + (((np.trace(Ginv, axis1=1, axis2=2)/h**2)*Kxy)[:,:,newaxis] * DxyGinv).sum(axis=1)\
                    - np.matmul( ((2/h**2)*Kxy[:,:,newaxis]*DxyGinv).sum(axis=0)[:,newaxis,:], Ginv ).reshape(M,D)

            vect /= M
            if optType == 'adag':
                if self.iter == 0:
                    self.historical_grad += vect ** 2
                else:
                    self.historical_grad = alpha * self.historical_grad + (1 - alpha) * (vect ** 2)
                        
                theta += self.stepsize * (vect / (self.fudge_factor + np.sqrt(self.historical_grad)))
            elif optType == 'gd':
                theta += self.stepsize * vect
            elif optType == 'sgd':
                theta += self.stepsize/pow(self.iter+1., alpha) * vect
            else: raise ValueError('unknown \'optType\': \'%s\'!'%optType)

        return theta

