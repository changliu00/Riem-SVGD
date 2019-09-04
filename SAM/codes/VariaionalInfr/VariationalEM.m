% function model = VariationalEM(model[, dict[, filePrefix]]]);
function model = VariationalEM(model, dict, filePrefix);

if nargin < 2
  dict = [];
end

if nargin < 3
  filePrefix = 'unknown_data_set';
end

SAVE_MODEL = 1;
CONVERGENCE_THRESH = 0.01;

% Define numerical update rules
vMuRule = struct('paramName', 'vMu', ...
                 'maxIterations', 100, ...
                 'learningRate', 1e-3, ... 
                 'useGradientScaling', 1, ...
                 'gradientMax', 1/sqrt(model.V)/10, ... 
                 'likelihoodFn', @L_vMu, ...
                 'derivFn', @DerivL_vMu, ...
                 'constraintFn', @(x)(NormalizeColumns(x)));
vAlphaRule = struct('paramName', 'vAlpha', ...
                    'maxIterations', 300, ...
                    'learningRate', 1e-2, ...
                    'useGradientScaling', 1, ...
                    'gradientMax', 0.05, ...
                    'likelihoodFn', @L_vAlpha, ...
                    'derivFn', @DerivL_vAlpha, ...
                    'constraintFn', @(x)(max(x, 1e-5)));
xiRule = struct('paramName', 'xi', ...
                'maxIterations', 100, ...
                'learningRate', 1e-2, ...
                'useGradientScaling', 1, ...
                'gradientMax', 1, ... 
                'likelihoodFn', @L_xi, ...
                'derivFn', @DerivL_xi, ...
                'constraintFn', @(x)(max(x, 1e-5)));
alphaRule = struct('paramName', 'alpha', ...
                   'maxIterations', 100, ...
                   'learningRate', 1e-2, ...
                   'useGradientScaling', 1, ...
                   'gradientMax', 0.1, ... 
                   'likelihoodFn', @L_alpha, ...
                   'derivFn', @DerivL_alpha, ...
                   'constraintFn', @(x)(max(x, 1e-5)));

                    
% Helper functions and structures for testing for convergence
isSmall = @(x)(max(max(x)) < CONVERGENCE_THRESH);
parameterNames = {'vAlpha', 'vMu', 'vM', 'M', 'xi', 'kappa0', 'kappa1', 'alpha'};

% Initialize some stuff
oldValues = struct();
converged = struct();

if ~isfield(model, 'iteration')
    model.iteration = 1;
end

% Go!
tic;
while 1 
  disp(sprintf('(Start of iteration %d)', model.iteration));
  % Save old parameter values
  for i = 1:length(parameterNames)
    name = parameterNames{i};
    oldValues.(name) = model.(name);
  end

  % Perform an EM iteration
  model = NumericalUpdate(model, vAlphaRule);  %model = VEMUpdateVAlpha(model);
  model = NumericalUpdate(model, vMuRule);     %model = VEMUpdateVMu(model);
  model = VEMUpdateVM(model);
  model = VEMUpdateM(model);
  
  model = NumericalUpdate(model, xiRule);
  model = NumericalUpdate(model, alphaRule);
  
  %model = VEMUpdateKappa1(model);
  %model = VEMUpdateXi(model);
  %model = VEMUpdateKappa0(model);
  %model = VEMUpdateAlpha(model);

  disp(sprintf('(End of iteration %d)', model.iteration));
  model.iteration = model.iteration + 1;  

  if SAVE_MODEL
    fprintf('Writing model to disk...');
    
    % Save the model
    save model.mat;
    
    % Save topics to a file in ascii format
    topics = model.vMu;
    save topics.txt topics -ascii;

    % Save highest- and lowest-weighted words from each topic for qualitative
    % analysis.  This requires that the caller provided a dictionary.
    if ~isempty(dict)
      fid = fopen('topicWords.txt', 'w');
      PrintTopics(model, dict, 10, 10, fid);
      fclose(fid);
    end

    % Save MAP estimates of the documents' topic proportions
    mapTopicProportions = (model.vAlpha - 1) * diag(1./(sum(model.vAlpha, 1) - model.T));
    save(sprintf('%s.map.vEM', filePrefix), 'mapTopicProportions', '-ascii');

    % Save expectations of documents' topic proportions
    meanTopicProportions = model.vAlpha * diag(1./sum(model.vAlpha, 1));
    save(sprintf('%s.mean.vEM', filePrefix), 'meanTopicProportions', '-ascii');

    clear meanTopicProportions mapTopicProportions topics;
    fprintf('Done.\n');
  end
  
  % Check convergence; loop over the parameters
  allParamsConverged = 1;
  for i = 1:length(parameterNames)  
    name = parameterNames{i};
    converged.(name) = isSmall(model.(name) - oldValues.(name));
    if ~converged.(name)
      allParamsConverged = 0;
    end
  end
  
  if allParamsConverged
    fprintf('convergence!\n');
    break;
  end
end
disp('Done');
toc;
