% function likelihood = L_alpha(model);
% Evaluates L_[alpha], the portion of the log-likelihood dependent on alpha
function likelihood = L_alpha(model);

D = model.D;
T = model.T;
V = model.V;

alpha = model.alpha;
alpha0 = sum(alpha);

psiVAlpha = psi(model.vAlpha);
psiVAlpha0s = psi(sum(model.vAlpha, 1));

likelihood = sum(sum( ((alpha - 1) * ones(1, D)) .* psiVAlpha )) - (alpha0 - T)*sum(psiVAlpha0s) + ...
             D*LogGamma(alpha0) - D*sum(LogGamma(alpha));


% The expression above should be equivalent to (and more efficient than)
% this old version:
% -----------------
% Sum over documents and the T elements of each alpha
%mySum = 0;
%for d = 1:D
%  vAlphaD(1:T) = model.vAlpha(1:T, d);
%  vAlphaD0 = sum(vAlphaD);
%  for i = 1:T
%    mySum = mySum + (alpha(i)-1)*(psi(vAlphaD(i)) - psi(vAlphaD0));
%  end
%end
%
%likelihood2 = ...
%  mySum + D*log(gamma(alpha0)) - D*sum(log(gamma(alpha)));
% abs(likelihood - likelihood2)