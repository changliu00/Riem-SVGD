% function deriv = DerivRhoXi(model);
function deriv = DerivRhoXi(model);

V = model.V;
xi = model.xi;

aXi = AvK(V, xi);
derivAXi = DerivAvK(V, xi);

vAlpha0s = sum(model.vAlpha, 1);
esns = ESquaredNorm_Batch(model);
derivESquaredNormXis = DerivESquaredNormXi(model);

vMuTimesVAlphaDotDoc = sum(model.vAlpha .* (model.vMu' * model.v), 1);

%for d = 1:model.D
%  derivESquaredNormXis(d) = DerivESquaredNormXi(model, d);
%end

%deriv = derivAXi * dot(vMuTimesVAlpha, vd) / vAlpha0 / sqrt(esn) ...
%       - aXi * dot(vMuTimesVAlpha, vd) / 2 / vAlpha0 / esn^(3/2) * derivESquaredNormXi;

deriv = derivAXi * vMuTimesVAlphaDotDoc ./ (vAlpha0s .* sqrt(esns)) ...
        - aXi/2 * vMuTimesVAlphaDotDoc ./ (vAlpha0s .* esns.^(3/2)) .* derivESquaredNormXis;
      
      
