% function grad = GradRhoVMu_Batch(model);
%
% Just like GradRhoVMu, except it computes the gradient with respect to each
% document AND topic.  This is way faster than calling GradRhoVMu over and over 
% again, but uses a ton (2000 lbs.) more memory.  Returns a V-by-T-by-D
% matrix.
%
% TODO: Make a version that computes the gradient for a subset of the
% documents.  The caller can partition the documents and call GradRhoVMU_Batch
% on each subset.  This will be less efficient, but will reduce the memory
% requirements.
%
% TODO: Make a test that verifies that this function does the same thing as
% GradRhoVMu.
function grad = GradRhoVMu_Batch(model, documentIndices);

if nargin < 2
  documentIndices = 1:model.D;
end
numDocs = length(documentIndices);

T = model.T;
D = model.D;
V = model.V;
xi = model.xi;

aXi = AvK(V, xi);
%esns = ESquaredNorm_Batch(model);
esns = ESquaredNorm_Batch(model, documentIndices);

%gradEsns = GradESquaredNorm_Batch(model);
gradEsns = GradESquaredNorm_Batch(model, documentIndices);

%for t = 1:T
%  gradEsns(:, t) = GradESquaredNorm(model, d, t);
%end
%grad = aXi/vAlphaD0 * (vAlphaD(t)*vd/sqrt(esn) - dot(model.vMu*vAlphaD, vd)/(2*esn^(3/2))* gradEsn);

vAlpha0s = sum(model.vAlpha, 1);
%vMuTimesVAlphaDotV = sum(model.vAlpha .* (model.vMu' * model.v), 1);
%grad = zeros(V, T, D);
%for d = 1:D

vMuTimesVAlphaDotV = sum(model.vAlpha(:, documentIndices) .* (model.vMu' * model.v(:, documentIndices)), 1);
grad = zeros(V, T, numDocs);
for i = 1:numDocs
  d = documentIndices(i);
  vAlphaD = model.vAlpha(:, d);
  %grad(:, :, d) = aXi/vAlpha0s(d) * (1/sqrt(esns(d))*model.v(:, d)*vAlphaD' - vMuTimesVAlphaDotV(d)/(2*esns(d)^(3/2))* gradEsns(:, :, d));
  grad(:, :, i) = aXi/vAlpha0s(d) * (1/sqrt(esns(i))*model.v(:, d)*vAlphaD' - vMuTimesVAlphaDotV(i)/(2*esns(i)^(3/2))* gradEsns(:, :, i));
end

%dot(vMu * vAlphaD, vd) = vAlphaD' * vMu' * vd  (1-by-T * (T-by-V * V-by-D)
%                       = element d of sum(vAlpha .* (vMu' * v), 1) 