% function deriv = DerivESquaredNorm_Batch(model);
function deriv = DerivESquaredNorm_Batch(model);
T = model.T;
V = model.V;
D = model.D;
xi = model.xi;
vAlpha = model.vAlpha;
vAlpha0s = sum(vAlpha, 1);

aXiSquared = AvK(V,xi)^2;

esns = ESquaredNorm_Batch(model);
vMuTimesVAlphaTimesVMu = model.vAlpha' * (model.vMu' * model.vMu);  % D by T

% deriv = zeros(T, D);
% for d = 1:D
%   for t = 1:T
%     deriv(t, d) = (1 + 2*(1-aXiSquared)*vAlpha(t, d) + 2*aXiSquared*vMuTimesVAlphaTimesVMu(d, t) ...
%                    - esns(d)*(2*vAlpha0s(d)+1)) / (vAlpha0s(d)*(vAlpha0s(d) + 1));
%   end
% end

perDocWeights = 1./(vAlpha0s .* (vAlpha0s + 1));
deriv = (1 + 2*(1-aXiSquared)*vAlpha + 2*aXiSquared*vMuTimesVAlphaTimesVMu');
deriv = bsxfun(@minus, deriv,  esns .* (2*vAlpha0s+1));
deriv = deriv * SparseDiag(perDocWeights);
