% function deriv = DerivESquaredNormXi(model,);
function deriv = DerivESquaredNormXi(model);
xi = model.xi;
V = model.V;
aXi = AvK(V, xi);
derivAXi = DerivAvK(V, xi);

vAlpha0s = sum(model.vAlpha, 1);
sumVAlphasSquared = sum(model.vAlpha.^2, 1);
%foo = diag(model.vAlpha' * (model.vMu' * model.vMu) * model.vAlpha)';
vMuVAlphaVMuVAlpha = sum((model.vAlpha' * (model.vMu' * model.vMu))' .* model.vAlpha, 1);
deriv = 2*aXi*derivAXi*(vMuVAlphaVMuVAlpha - sumVAlphasSquared) ./ (vAlpha0s .* (vAlpha0s + 1));