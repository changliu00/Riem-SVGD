import numpy as np
from numpy import newaxis
import scipy.io
from sklearn.model_selection import train_test_split
import numpy.matlib as nm
from rsvgd import RSVGD

class RSVGD_BayesianLR_simp:
    def __init__(self, X, Y, M, alpha = 0.01):
        self.X, self.Y = X, Y # Y\in{0,1}. X: N*D; Y: N
        # self.batchsize = min(batchsize, X.shape[0])
        self.M = M
        self.alpha = alpha

        self.N, self.D = X.shape
        # self.permutation = np.random.permutation(self.N)
        # self.iter = 0

        self.Ginv = np.zeros([M, self.D, self.D])
        self.gradDet = np.zeros([M, self.D])
        self.gradGinv = np.zeros([M, self.D])

    def update_info(self, theta): # theta: M*D
        if self.M != theta.shape[0]: raise ValueError('NumParticle not fit!')
        w = theta
        X = self.X
        r = np.exp(w.dot(X.T)) # M*N
        invr = 1/(1+r)
        c = r * invr**2 # M*N
        f = (2*invr - 1) * c # M*N

        gradp = -w/self.alpha + (invr + self.Y-1).dot(X) # M*D
        for m in range(self.M):
            G = (c[m,]*X.T).dot(X) + np.eye(D)/self.alpha
            self.Ginv[m,...] = np.linalg.inv(G)
            Ginvm = self.Ginv[m,...]
            self.gradDet[m,] = ((X.dot(Ginvm) * X).sum(axis=1) * f[m,]).dot(X)
            self.gradGinv[m,] = -self.gradDet[m,].dot(Ginvm)
            # gradG = np.matmul( (X.T*f[m,])[newaxis,:,:] * X.T[:,newaxis,:], X ) # D*D*D
            # self.gradDet[m,] = (Ginvm * gradG).sum(axis=(1,2))
            # self.gradGinv[m,] = - (np.matmul(gradG, Ginvm[:,:,newaxis]).sum(axis=0).reshape(D)).dot(Ginvm)

        return gradp, self.Ginv, self.gradDet, self.gradGinv

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
        prob = np.exp(y_test[:,newaxis]*inprods) / (1 + np.exp(inprods))
        prob = np.mean(prob, axis=1)
        acc = np.mean(prob > 0.5)
        llh = np.mean(np.log(prob))
        return [acc, llh] 

if __name__ == '__main__':
    # dataid = 'large'; alpha = 1e-2; M = 100; batchsize = -1; stepsize = 5e-2; optAlpha = 0.9; optType = 'gd';
    # bandwidth = [1e-4, 1e-2, 1e-1, 1, 10, 30, 100, 300, 1000]; n_iter = 10; n_round = 40;
    # dataid = 'small'; alpha = 1e-2; M = 100; batchsize = -1; stepsize = 5e-2; optAlpha = 0.9; optType = 'gd';
    dataid = 'small'; alpha = 1e-2; M = 100; batchsize = -1; stepsize = 3e-1; optAlpha = 0.9; optType = 'gd';
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
        app += 1; filename = 'res_%s_lrSimp_rsvgd_%s_%d.txt'%(dataName, optType, app)
        try: fid = open(filename, 'r')
        except IOError: break
        fid.close()

    with open(filename, 'w') as fid:
        fid.write('alpha=%.2e; '%alpha + 'M=%d; '%M + 'batchsize=%d; '%batchsize + 'stepsize=%.2e; '%stepsize + 'optAlpha=%.2e; '%optAlpha + 'bandwidth=%s;'%str(bandwidth) + '\n')
        fid.write('iter accuracy log-llh\n')

    # initialization
    theta = np.random.normal(0, np.sqrt(alpha), [M, D])
    model = RSVGD_BayesianLR_simp(X_train, y_train, M, alpha)
    rsvgdobj = RSVGD(theta, model.update_info, bandwidth, stepsize, optAlpha, optType)
    with open(filename, 'a') as fid:
        for it in range(n_round):
            theta = rsvgdobj.update(n_iter) #, debug=True)
            res = model.evaluation(theta, X_test, y_test)
            fid.write('%4d %.6f %.6e\n'%((it+1)*n_iter, res[0], res[1]))

