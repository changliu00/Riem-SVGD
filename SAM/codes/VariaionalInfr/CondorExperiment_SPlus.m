% function model = CondorExperiment_SPlus(T, dataFile, dictFile, filePrefix);
function model = CondorExperiment_SPlus(T, dataFile, dictFile, filePrefix);

% note that callers should set OCTAVE_PATH appropriately
% to include the tools and <matlab|octave>-host directories
% (or MATLABPATH in the matlab case)

% Initialize random seed using the system time
if ~IsOctave()
  rand('twister',sum(100*clock));
end

[documents, dict] = LoadCorpus(dataFile, dictFile); 

try
    disp('attempting to load model from disk...');
    load model;
    disp('...succeeded; starting from checkpoint');
catch load_error
    disp('...failed; initializing from scratch');
    model = VEMInit_NoVMF(documents, T);
end

% do variational EM!
model = VariationalEM_SPlus(model, dict, filePrefix);

