addpath('tools');

if IsOctave()
    addpath('octave-host')
else
    addpath('matlab-host')  % pull in psi_n, etc.
end

T = 10;  % Number of topics

dataFile = '../data/20news.train.sp';
dictFile = '../data/20news.train.V';

% Keep about 1000 documents from 20news
[documents, dict] = LoadCorpus(dataFile, dictFile, 1000, 0.5, 0.001);

model = VEMInit_NoVMF(documents, T);
model = VariationalEM(model, dict);
