% function esn = ESquaredNorm(model, d);
function esn = ESquaredNorm(model, d);
T = model.T;
xi = model.xi;
V = model.V;
vAlphaD = model.vAlpha(1:T, d);
vAlphaD0 = sum(vAlphaD);

aXiSquared = AvK(V,xi)^2;
vMuTimesVAlpha = model.vMu * vAlphaD;

%topicSquaredNorms = sum(model.vMu.^2, 1); % For Num_Deriv_ correctness
%esn = (vAlphaD0 + (1-aXiSquared)*sum(vAlphaD.^2 .* topicSquaredNorms') ...
%        + aXiSquared*(vMuTimesVAlpha'*vMuTimesVAlpha)) ...
%        / (vAlphaD0 * (vAlphaD0 + 1));
esn = (vAlphaD0 + (1-aXiSquared)*sum(vAlphaD.^2) ...
        + aXiSquared*(vMuTimesVAlpha'*vMuTimesVAlpha)) ...
        / (vAlphaD0 * (vAlphaD0 + 1));
