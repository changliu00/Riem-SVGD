% function grad = DerivL_vMu(model);
function grad = DerivL_vMu(model);
T = model.T;
V = model.V;
xi = model.xi;
V = model.V;
D = model.D;
k0 = model.kappa0;
k1 = model.kappa1;

%sumOverDocuments = zeros(V, T);
%for d = 1:D
%  sumOverDocuments = sumOverDocuments + GradRhoVMu_BatchT(model, d);
%end
sumOverDocuments = ComputeSumOverDocs(model);

% New: vM is a V-by-1 vector, sumOverDocuments is a V-by-T matrix
% Add the first argument to each column of the second argument
grad = bsxfun(@plus, AvK(V,xi) * AvK(V,k0) * xi * model.vM, k1 * sumOverDocuments);

% Project the gradients into the tangent space at each topic
for t = 1:T
  gradForMuT = grad(1:V, t);
  vMuT = model.vMu(1:V, t);
  grad(1:V, t) = gradForMuT - vMuT*(vMuT'*gradForMuT);
end
return;


function s = ComputeSumOverDocs(model);
T = model.T;
D = model.D;
V = model.V;
xi = model.xi;
aXi = AvK(V, xi);
aXiSquared = AvK(model.V, model.xi)^2;
    
esns = zeros(D, 1);  
for d = 1:D
  esns(d) = ESquaredNorm(model, d);
end
vAlpha0s = sum(model.vAlpha, 1)';  % Column vector


% For single d:  aXi/vAlphaD0 /sqrt(esn) * vd * vAlphaD' 
firstTerm = model.v * (model.vAlpha * SparseDiag(aXi ./ (vAlpha0s .* sqrt(esns))))';

% Weights per document for everything that was in GradESN; from second term in GradRhoVMu_BatchT
% For a single d: aXi/vAlphaD0 / (2*esn^(3/2)) * dot(model.vMu*vAlphaD, vd)
perDocWeights = aXi ./ vAlpha0s ./ (2*esns.^(3/2)) .* sum(model.vAlpha .* (model.vMu' * model.v), 1)';

secondTermDocWeights = 1./(vAlpha0s.*(vAlpha0s+1))*(2*(1-aXiSquared));
secondTerm = sum(perDocWeights .* secondTermDocWeights) * model.vMu;

% Last term in GradESN times per-doc factors from GradRhoVMu...
thirdTermDocWeights = 1./(vAlpha0s.*(vAlpha0s+1)) * 2*aXiSquared;  % From GradESN
rescaledVAlphas = model.vAlpha * SparseDiag(sqrt(perDocWeights .* thirdTermDocWeights));
thirdTerm = model.vMu * (rescaledVAlphas*rescaledVAlphas');

s = firstTerm - secondTerm - thirdTerm;
return;
