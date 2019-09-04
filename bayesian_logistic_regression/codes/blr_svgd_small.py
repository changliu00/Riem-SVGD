import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
import numpy.matlib as nm
from svgd import SVGD

'''
    Example of Bayesian Logistic Regression (the same setting as Gershman et al. 2012):
    The observed data D = {X, y} consist of N binary class labels, 
    y_t \in {-1,+1}, and d covariates for each datapoint, X_t \in R^d.
    The hidden variables \theta = {w, \alpha} consist of d regression coefficients w_k \in R,
    and a precision parameter \alpha \in R_+. We assume the following model:
        p(\alpha) = Gamma(\alpha; a, b)
        p(w_k | a) = N(w_k; 0, \alpha^-1)
        p(y_t = 1| x_t, w) = 1 / (1+exp(-w^T x_t))
'''
class BayesianLR_simp:
    def __init__(self, X, Y, batchsize=100, alpha=0.01):
        self.X, self.Y = X, Y
        # TODO. Y in \in{+1, -1}
        self.batchsize = min(batchsize, X.shape[0])
        self.alpha = alpha
        
        self.N = X.shape[0]
        self.permutation = np.random.permutation(self.N)
        self.iter = 0
    
        
    def dlnprob(self, theta):
        
        if self.batchsize > 0:
            batch = [ i % self.N for i in range(self.iter * self.batchsize, (self.iter + 1) * self.batchsize) ]
            ridx = self.permutation[batch]
            self.iter += 1
        else:
            ridx = np.random.permutation(self.X.shape[0])
            
        Xs = self.X[ridx, :]
        Ys = self.Y[ridx]
        
        w = theta
        r = np.exp(w.dot(Xs.T)) # M*N
        invr = 1/(1+r)
        gradp = -w/self.alpha + (invr + Ys-1).dot(Xs) * self.X.shape[0]/Xs.shape[0] # M*D
        return gradp  # % first order derivative 
    
    def evaluation0(self, theta, X_test, y_test):
        y = y_test.copy()
        y[y == 0] = -1
        M, n_test = theta.shape[0], len(y)

        prob = np.zeros([n_test, M])
        for t in range(M):
            coff = np.multiply(y, np.sum(-1 * np.multiply(nm.repmat(theta[t, :], n_test, 1), X_test), axis=1))
            prob[:, t] = np.divide(np.ones(n_test), (1 + np.exp(coff)))
        
        prob = np.mean(prob, axis=1)
        acc = np.mean(prob > 0.5)
        llh = np.mean(np.log(prob))
        return [acc, llh]

    def evaluation(self, theta, X_test, y_test):
        w = theta
        inprods = X_test.dot(w.T)
        prob = np.exp(y_test[:,np.newaxis]*inprods) / (1 + np.exp(inprods))
        prob = np.mean(prob, axis=1)
        acc = np.mean(prob > 0.5)
        llh = np.mean(np.log(prob))
        return [acc, llh] 

if __name__ == '__main__':
    # dataid = 'large'; alpha = 1e-2; M = 100; batchsize = -1; stepsize = 1e-3; optAlpha = 0.9; optType = 'adag';
    # bandwidth = -1; n_iter = 10; n_round = 40;
    # dataid = 'small'; alpha = 1e-2; M = 100; batchsize = -1; stepsize = 1e-3; optAlpha = 0.9; optType = 'adag';
    # bandwidth = -1; n_iter = 10; n_round = 40;
    dataid = 'small'; alpha = 1e-2; M = 100; batchsize = -1; stepsize = 3e-3; optAlpha = 0.9; optType = 'adag';
    bandwidth = [1e-4, 1e-2, 1e-1, 1, 10, 30, 100, 300, 1000]; n_iter = 10; n_round = 40;
    if dataid == 'large':
        data = scipy.io.loadmat('../data/covertype.mat')
        X_input = data['covtype'][:, 1:]
        y_input = data['covtype'][:, 0]
        y_input[y_input == 2] = 0
        # split the dataset into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2, random_state=42)
        dataName = 'covtype'
    elif dataid == 'small':
        data = scipy.io.loadmat('../data/benchmarks.mat')
        #                                           # totSz  Tr  Ts  Feat SplitSz
        #splitIdx = 42; dataName =        'banana'; # 5300  400 4900    2 100
        #splitIdx = 42; dataName = 'breast_cancer'; #  263  200   77    9 100
        #splitIdx = 42; dataName =      'diabetis'; #  768  468  300    8 100
        #splitIdx = 42; dataName =   'flare_solar'; #  144  666  400    9 100
        #splitIdx = 42; dataName =        'german'; # 1000  700  300   20 100
        #splitIdx = 42; dataName =         'heart'; #  270  170  100   13 100
        #splitIdx = 19; dataName =         'image'; # 2086 1300 1010   18  20
        #splitIdx = 42; dataName =      'ringnorm'; # 7400  400 7000   20 100
        splitIdx = 19; dataName =        'splice'; # 2991 1000 2175   60  20
        #splitIdx = 42; dataName =       'thyroid'; #  215  140   75    5 100
        #splitIdx = 42; dataName =       'titanic'; #   24  150 2051    3 100
        #splitIdx = 42; dataName =       'twonorm'; # 7400  400 7000   20 100
        #splitIdx = 42; dataName =      'waveform'; # 5000  400 4600   21 100
        X_train = data[dataName]['x'][0,0][ data[dataName]['train'][0,0][splitIdx-1,:]-1, : ]
        y_train = data[dataName]['t'][0,0][ data[dataName]['train'][0,0][splitIdx-1,:]-1, : ]
        X_test = data[dataName]['x'][0,0][ data[dataName]['test'][0,0][splitIdx-1,:]-1, : ]
        y_test = data[dataName]['t'][0,0][ data[dataName]['test'][0,0][splitIdx-1,:]-1, : ]
        y_train = y_train.ravel(); y_test = y_test.ravel();
        y_train[y_train == -1] = 0; y_test[y_test == -1] = 0;
    ######################

    X_train = np.hstack([X_train, np.ones([X_train.shape[0], 1])])
    X_test = np.hstack([X_test, np.ones([X_test.shape[0], 1])])
    D = X_train.shape[1]

    app = 0
    while True:
        app += 1; filename = 'res_%s_lrSimp_svgd_%s_%d.txt'%(dataName, optType, app)
        try: fid = open(filename, 'r')
        except IOError: break
        fid.close()

    with open(filename, 'w') as fid:
        fid.write('alpha=%.2e; '%alpha + 'M=%d; '%M + 'batchsize=%d; '%batchsize + 'stepsize=%.2e; '%stepsize + 'optAlpha=%.2e; '%optAlpha + 'bandwidth=%s;'%str(bandwidth) + '\n')
        fid.write('iter accuracy log-llh\n')

    # initialization
    theta = np.random.normal(0, np.sqrt(alpha), [M, D])
    model = BayesianLR_simp(X_train, y_train, batchsize=batchsize, alpha=alpha) # batchsize = 100
    svgdobj = SVGD(theta, model.dlnprob, bandwidth, stepsize, optAlpha, optType)
    with open(filename, 'a') as fid:
        for it in range(n_round):
            theta = svgdobj.update(n_iter) #, debug=True)
            res = model.evaluation(theta, X_test, y_test)
            fid.write('%4d %.6f %.6e\n'%((it+1)*n_iter, res[0], res[1]))

