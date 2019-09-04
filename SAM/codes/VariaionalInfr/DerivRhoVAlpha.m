% function deriv = DerivRhoVAlpha(model, d, t);
function deriv = DerivRhoVAlpha(model, d, t);

T = model.T;
xi = model.xi;
V = model.V;
vAlphaD = model.vAlpha(1:T, d);
vAlphaD0 = sum(vAlphaD);

aXi = AvK(V,xi);
vMuTimesVAlpha = model.vMu * vAlphaD;

esn = ESquaredNorm(model, d);

firstTerm = (model.vMu(1:V, t)/vAlphaD0 - vMuTimesVAlpha/vAlphaD0^2)/sqrt(esn);
secondTerm = -vMuTimesVAlpha/vAlphaD0 / (2*esn^(3/2)) * DerivESquaredNorm(model, d, t);

deriv = aXi * dot((firstTerm + secondTerm), model.v(:, d));