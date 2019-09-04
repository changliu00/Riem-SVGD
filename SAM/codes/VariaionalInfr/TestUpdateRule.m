% function TestUpdateRule(model, rule);
function TestUpdateRule(model, rule, verbose);

if nargin < 3
  verbose = 1;
end

DELTA = 1e-5;
TOLERANCE = 1e-3;

paramName = rule.paramName;
derivFn = rule.derivFn;
likeFn = rule.likelihoodFn;

origParamValue = model.(paramName);
origLikelihood = likeFn(model);

[numRows, numCols] = size(model.(paramName));

closedFormDerivative = derivFn(model);
numericalDerivative = zeros(numRows, numCols);
for row = 1:numRows
  for col = 1:numCols
    model.(paramName)(row, col) = origParamValue(row, col) + DELTA;
    numDeriv = (likeFn(model) - origLikelihood)/DELTA;
    model.(paramName)(row, col) = origParamValue(row, col);
    
    numericalDerivative(row, col) = numDeriv;
    maxError = abs(numDeriv - closedFormDerivative(row, col));
    if maxError > TOLERANCE
      fprintf('%s(%d, %d):  max gradient error is %f (closed form = %f, numerical = %f)\n', paramName, row, col, maxError, closedFormDerivative(row, col), numDeriv);
    end
    
    if verbose
      fprintf('.');
    end
  end
end

if verbose
  fprintf('done.\n');
end

maxError = max(max(abs(closedFormDerivative - numericalDerivative)));
if maxError > TOLERANCE
  fprintf('%s:  max gradient error is %f\n', paramName, maxError);
else
  fprintf('%s:  PASS\n', paramName);
end
