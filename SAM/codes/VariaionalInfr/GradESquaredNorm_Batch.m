% function grad = GradESquaredNorm(model);
% Returns (V, T, D)-dim matrix
%
% TODO: Make a version that computes the gradient for subset of the
% documents -- this will reduce the memory requirements.
%
% TODO: Make a test that verifies that this function does the same thing as
% GradESquaredNorm.
function grad = GradESquaredNorm_Batch(model, documentIndices);

if nargin > 2
  error('Usage: GradESquaredNorm_Batch(model)');
end

if nargin < 2
  documentIndices = 1:model.D;
end
numDocs = length(documentIndices);

T = model.T;
D = model.D;
V = model.V;
xi = model.xi;

aXiSquared = AvK(model.V, model.xi)^2;

%vAlpha0s = sum(model.vAlpha, 1);  % One entry per document
%vMuTimesVAlpha = model.vMu * model.vAlpha; % One column per document
vMuTimesVAlpha = model.vMu * model.vAlpha(:, documentIndices); % One column per document
firstTerm = 2*(1-aXiSquared)*model.vMu;

% grad = zeros(V, T, D);
grad = zeros(V, T, numDocs);
%for d = 1:D
for i = 1:numDocs
  d = documentIndices(i);
  vAlphaD = model.vAlpha(1:T, d);
  vAlphaD0 = sum(vAlphaD);
  %secondTerm = (2*aXiSquared) * vMuTimesVAlpha(:, d) * vAlphaD';
  %grad(:, :, d) = 1/(vAlphaD0*(vAlphaD0+1)) * (firstTerm + secondTerm);
  secondTerm = (2*aXiSquared) * vMuTimesVAlpha(:, i) * vAlphaD';
  grad(:, :, i) = 1/(vAlphaD0*(vAlphaD0+1)) * (firstTerm + secondTerm);
end