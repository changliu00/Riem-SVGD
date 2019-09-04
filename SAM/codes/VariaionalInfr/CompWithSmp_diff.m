% script for comparison with sampling methods

T = 20;
beginTime = 500;
timeInterv = 1000;
max_iter = 200000;
dataFile = '../../data/20News-diff/diff.train.tfidf1.data';
addpath('.');
addpath('./tools');
addpath('./matlab-host');

model = struct();
model = LoadData_CompWithSmp(model, dataFile);
model = VEMInit_NoVMF_CompWithSmp_diff(model, T, max_iter);
model = VariationalEM_CompWithSmp(model, 'diff_', beginTime, timeInterv);

