% function deriv = DerivL_vAlpha(model);
% Returns a matrix containing component-wise derivatives of 
%   d/d{\tilde{\alpha}_d,i} L_{vAlpha_{d,i}}
function deriv = DerivL_vAlpha(model);

V = model.V;
D = model.D;
T = model.T;
alpha = model.alpha;
kappa1 = model.kappa1;
alpha0 = sum(alpha);

psiVAlpha = psi(model.vAlpha);
derivPsiVAlpha = psi_n(1, model.vAlpha);
gammaVAlpha = gamma(model.vAlpha);

vAlpha0s = sum(model.vAlpha, 1);  % Sum of each document's vAlpha
gammaVAlpha0s = gamma(vAlpha0s);
psiVAlpha0s = psi(vAlpha0s);
derivPsiVAlpha0s = psi_n(1, vAlpha0s);

%derivsOfRho = zeros(T, D);
%for d = 1:D
%  for t = 1:T
%    derivsOfRho(t, d) = DerivRhoVAlpha(model, d, t);
%  end
%end
derivsOfRho = DerivRhoVAlpha_Batch(model); % Compute derivs w.r.t. all vAlphas
 
%deriv = zeros(T, D);
%for d = 1:D
%  for t = 1:T
%    deriv(t, d) = ...
%      (alpha(t)-1)*derivPsiVAlpha(t, d) ...  
%      - derivPsiVAlpha0s(d)*(alpha0 - T) ... 
%      + kappa1*derivsOfRho(t, d) ... 
%      - (model.vAlpha(t, d) - 1)*derivPsiVAlpha(t, d) - psiVAlpha(t, d) ...
%      + derivPsiVAlpha0s(d)*(vAlpha0s(d) - T) + psiVAlpha0s(d) ...
%      - psiVAlpha0s(d) + psiVAlpha(t, d);
%      %+ psiVAlpha0s(d) - psiVAlpha(t, d); % 3/19 fix
%  end
%end

% Deriv is T-by-D
deriv = SparseDiag(alpha - 1) * derivPsiVAlpha ...  % Scale rows by alpha(t)-1
        + kappa1*derivsOfRho ...
        - (model.vAlpha - 1).*derivPsiVAlpha - psiVAlpha ...
        + psiVAlpha;

addToEachRow = -derivPsiVAlpha0s*(alpha0 - T) ...
               + derivPsiVAlpha0s .* (vAlpha0s - T); 

deriv = bsxfun(@plus, deriv, addToEachRow);



% This is equivalent, and will probably be much faster once if we can
% replace the repmat calls with bsxfun.
%deriv2 = kappa1 * derivsOfRho ...
%         + repmat(derivPsiVAlpha0s .* (vAlpha0s - alpha0), T, 1) ...
%         - derivPsiVAlpha .* (model.vAlpha - repmat(model.alpha, 1, D));
