% function derivEsn = DerivESquaredNorm(model, d, j);
function derivEsn = DerivESquaredNorm(model, d, j);
T = model.T;
V = model.V;
xi = model.xi;
vAlphaD = model.vAlpha(1:T, d);
vAlphaD0 = sum(vAlphaD);

aXiSquared = AvK(V,xi)^2;
vMuTimesVAlpha = model.vMu * vAlphaD;

esn = ESquaredNorm(model, d);
derivEsn = (1 + 2*(1-aXiSquared)*vAlphaD(j) + 2*aXiSquared*(vMuTimesVAlpha'*model.vMu(1:V, j)) ...
           - esn*(2*vAlphaD0+1)) / (vAlphaD0*(vAlphaD0 + 1));
