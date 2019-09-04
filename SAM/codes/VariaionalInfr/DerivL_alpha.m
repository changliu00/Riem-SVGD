% function deriv = DerivL_alpha(model);
% Returns the gradient of L_alpha w.r.t. alpha.
function deriv = DerivL_alpha(model);
D = model.D;
kappa1 = model.kappa1;
alpha0 = sum(model.alpha);
vAlpha0s = sum(model.vAlpha, 1);  % vAlpha0 for each document

deriv = sum(psi(model.vAlpha), 2) - sum(psi(vAlpha0s)) ...
        + D*psi(alpha0) - D*psi(model.alpha);
