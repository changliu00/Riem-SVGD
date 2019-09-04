% function esns = ESquaredNorm_Batch(model);
function esns = ESquaredNorm_Batch(model);

T = model.T;
xi = model.xi;
V = model.V;
D = model.D;


aXiSquared = AvK(V,xi)^2;
%vMuVAlphaVMuVAlpha = diag(model.vAlpha' * (model.vMu' * model.vMu) * model.vAlpha)';
vMuVAlphaVMuVAlpha = sum((model.vAlpha' * (model.vMu' * model.vMu))' .* model.vAlpha, 1);
vAlpha0s = sum(model.vAlpha, 1);
vAlphaSquares = sum(model.vAlpha.^2, 1);

% esns = zeros(1, D);
% for d = 1:D
%   vAlphaD = model.vAlpha(1:T, d);  % The vAlpha column for this doc
%   %esns(d) = (vAlphaD0s(d) + (1-aXiSquared)*vAlphaSquares(d) ...
%   %          + aXiSquared*vAlphaD'*vMuTimesVMu*vAlphaD) / (vAlphaD0s(d) * (vAlphaD0s(d)  + 1));
%   esns(d) = (vAlphaD0s(i) + (1-aXiSquared)*vAlphaSquares(i) ...
%             + aXiSquared*vAlphaD'*vMuTimesVMu*vAlphaD) / (vAlphaD0s(i) * (vAlphaD0s(i) + 1));
% end

esns = (vAlpha0s + (1-aXiSquared)*vAlphaSquares + aXiSquared*vMuVAlphaVMuVAlpha) ./ (vAlpha0s .* (vAlpha0s + 1));