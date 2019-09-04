addpath('tools');

if IsOctave()
    addpath('octave-host');
else
    addpath('matlab-host');  % pull in psi_n, etc.
end

T = 10;  % Number of topics

dataFile = '../data/nips-452D-3238V-0.900-0.001-20.evidence.sp';
dictFile = '../data/nips-452D-3238V-0.900-0.001-20.evidence.V';
dict = LoadDictionary(dictFile);
model = VEMInit(dataFile, T);
model = VariationalEM(model);
