% function model = NumericalUpdate(model, updateRule);
function model = NumericalUpdate(model, updateRule);
GRADIENT_CONVERGENCE_THRESH = 0.01;
LIKELIHOOD_CONVERGENCE_THRESH = 0.05;
LEARNING_RATE_BACKOFF = 2;

learningRate = getParam(updateRule, 'learningRate');
likelihoodFn = getParam(updateRule, 'likelihoodFn');
derivFn = getParam(updateRule, 'derivFn');
paramName = getParam(updateRule, 'paramName');

% Process optional fields in updateRule
constraintFn = getOptionalParam(updateRule, 'constraintFn', @(x)(x));
useGradientScaling = getOptionalParam(updateRule, 'useGradientScaling', 1);
gradientMax = getOptionalParam(updateRule, 'gradientMax', 1);
adaptInterval = getOptionalParam(updateRule, 'adaptInterval', 10);
printInterval = getOptionalParam(updateRule, 'printInterval', 10);
maxIterations = getOptionalParam(updateRule, 'maxIterations', 500);
verbose = getOptionalParam(updateRule, 'verbose', 1);

converged = 0;
iteration = 1;
likelihoodHistory = [likelihoodFn(model)];
oldParamValue = model.(paramName);
while ~converged
  % Compute gradient
  gradient = derivFn(model);
  maxGradient = max(max(abs(gradient)));
  
  % If gradient scaling is enabled, we'll scale the gradient so that the
  % largest element is equal to the learning rate
  change = gradient * learningRate;
  maxChange = maxGradient * learningRate; 
  if useGradientScaling && (maxChange > gradientMax)
    change = change * (gradientMax / maxChange); 
    %fprintf('maxchange = %f, scaling to be %f\n', maxChange, max(max(change));
    %change = gradient / max(max(abs(gradient))) * learningRate;
  end

  % Update the parameter
  model.(paramName) = model.(paramName) + change;
  % Enforce constraints on the model parameter
  model.(paramName) = constraintFn(model.(paramName));
    
  % Every few iterations, check that the likelihood lower bound has gone up.  
  % If it has decreased, roll back to the previous model and decrease the 
  % learning rate.
  if mod(iteration, adaptInterval) == 0
    newLikelihood = likelihoodFn(model);
    likelihoodChange = newLikelihood - likelihoodHistory(end);
    if likelihoodChange < 0
      learningRate = learningRate / LEARNING_RATE_BACKOFF;
      model.(paramName) = oldParamValue;  % Revert to old val
      iteration = iteration - adaptInterval + 1;
      disp(sprintf('(%s) Likelihood decreased by %.2x; lowering learning rate to %.2x and rolling back to iteration %d', paramName, likelihoodChange, learningRate, iteration));
      continue;
    else
      likelihoodHistory(end+1) = newLikelihood;
      oldParamValue = model.(paramName);
    end


  end

  % Print an update every few iterations
  if verbose && (mod(iteration, printInterval) == 0)
    if length(likelihoodHistory) >= 2
      likelihoodChange = (likelihoodHistory(end) - likelihoodHistory(end-1)) / adaptInterval;
      disp(sprintf('(%s iter %d): max gradient = %f, likelihood change = %f', paramName, iteration, maxGradient, likelihoodChange));
    end
  end

  %convergedInLikelihood = (length(likelihoodHistory) > 1) && ...
  %  (likelihoodHistory(end) - likelihoodHistory(end-1) < LIKELIHOOD_CONVERGENCE_THRESH);
  %if convergedInLikelihood
  %  disp(sprintf('(%s) Converged in likelihood', paramName));
  %  converged = 1;
  %end

  % Check for convergence
  convergedInGradient = (maxGradient < GRADIENT_CONVERGENCE_THRESH);
  if convergedInGradient 
    disp(sprintf('(%s) Converged in gradient', paramName));
    converged = 1;
  end


  if iteration == maxIterations
    disp(sprintf('(%s) Bailing out; reached iteration limit', paramName));
    converged = 1;
  end
  iteration = iteration + 1;
end
disp(sprintf('(%s) Done', paramName));
return;


function result = getParam(params, paramName);
if isfield(params, paramName)
  result = params.(paramName);
else
  error(sprintf('Update rule structure must contain a "%s" field', paramName));
end

function result = getOptionalParam(params, paramName, defaultVal);
if isfield(params, paramName)
  result = params.(paramName);
else
  result = defaultVal;
end
