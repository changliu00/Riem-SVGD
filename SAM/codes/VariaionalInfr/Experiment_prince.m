addpath('tools');

if IsOctave()
    addpath('octave-host')
else
    addpath('matlab-host')  % pull in psi_n, etc.
end

T = 50;  % Number of topics

dataFile = '../data/prince-paragraph-267D-1519V-0.200-0.000-1.evidence.sp'
dictFile = '../data/prince-paragraph-267D-1519V-0.200-0.000-1.evidence.V'

[documents, dict] = LoadCorpus(dataFile, dictFile); 

model = VEMInit_NoVMF(documents, T);
model = VariationalEM(model, dict);
