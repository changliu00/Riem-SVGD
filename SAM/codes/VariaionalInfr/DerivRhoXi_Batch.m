% function deriv = DerivRhoXi_Batch(model);
function deriv = DerivRhoXi_Batch(model);

V = model.V;
xi = model.xi;

aXi = AvK(V, xi);
derivAXi = DerivAvK(V, xi);

esns = ESquaredNorm_Batch(model, d);

vMuTimesVAlpha = model.vMu * model.vAlpha;
vAlpha0s = sum(model.vAlpha, 1);

deriv = zeros(1, d);
derivsOfESquaredNormXi = zeros(1, d);
for d = 1:D
  derivsOfESquaredNormXi(d) = DerivESquaredNormXi(model, d);
  
  deriv(d) = derivAxi * dot(vMuTimesVAlpha(:, d), model.v(:, d)) / vAlpha0s(d) / sqrt(esns(d)) ...
        - aXi * dot(vMuTimesVAlpha(:, d), model.v(:, d)) / 2 / vAlpha0s(d) / esns(d)^(3/2) * derivsOfESquaredNormXi(d);
end

