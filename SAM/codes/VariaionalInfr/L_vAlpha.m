% function like = L_vAlpha(model);
function like = L_vAlpha(model);

V = model.V;
global D;
global T;
D = model.D;
T = model.T;
alpha = model.alpha;
kappa1 = model.kappa1;
alpha0 = sum(alpha);

psiVAlpha = psi(model.vAlpha);
derivPsiVAlpha = psi_n(1, model.vAlpha);

vAlpha0s = sum(model.vAlpha, 1);  % Sum of each document's vAlpha
psiVAlpha0s = psi(vAlpha0s);
derivPsiVAlpha0s = psi_n(1, vAlpha0s);

alphaMinusOneMatrix = (alpha - 1) * ones(1, D);

%sumOfRhos = 0;
%for d = 1:D
%  sumOfRhos = sumOfRhos + rho(model, d);
%end
sumOfRhos = sum(Rho_Batch(model));

like = sum(sum(alphaMinusOneMatrix .* psiVAlpha)) - (alpha0 - T)*sum(psiVAlpha0s) ...
       + D*LogGamma(alpha0) - D*sum(LogGamma(alpha)) ...
       + kappa1 * sumOfRhos ...
       - sum(sum((model.vAlpha - 1) .* psiVAlpha)) + sum(psiVAlpha0s .* (vAlpha0s - T)) ...
       - sum(LogGamma(vAlpha0s)) + sum(sum(LogGamma(model.vAlpha)));
       %+ sum(LogGamma(vAlpha0s)) - sum(sum(LogGamma(model.vAlpha)));  % 3/19 fix
    
return;    


%error('Barf');
%like = 0;    
%for d = 1:D
%  like = like + ...
%    sum((alpha-1) .* psiVAlpha(:, d)) - psiVAlpha0s(d)*sum(alpha-1) ...
%    + log(gamma(alpha0)) - sum(log(gamma(alpha))) ...
%    + kappa1 * rho(model, d) ...
%    - sum((model.vAlpha(:, d) - 1) .* psiVAlpha(:, d)) ...
%    + psiVAlpha0s(d)*sum((model.vAlpha(:, d) - 1)) ...
%    + log(gamma(vAlpha0s(d))) - sum(log(gammaVAlpha(:, d)));
%end
       
term1Fn = @(t, d)((alpha(t)-1)*psiVAlpha(t, d));
term2Fn = @(t, d)(-(alpha(t)-1)*psiVAlpha0s(d));
term3Fn = @(t, d)(log(gamma(alpha0)));
term4Fn = @(t, d)(-log(gamma(alpha(t))));
term5Fn = @(t, d)(kappa1*rho(model, d));
term6Fn = @(t, d)(-(model.vAlpha(t, d) - 1) * psiVAlpha(t, d));
term7Fn = @(t, d)((model.vAlpha(t, d) - 1) * psiVAlpha0s(d));
term8Fn = @(t, d)(log(gamma(vAlpha0s(d))));
term9Fn = @(t, d)(-log(gamma(model.vAlpha(t, d))));

cfLikeTerms = [ ...
  sum(sum(alphaMinusOneMatrix .* psiVAlpha)), ...
  - (alpha0 - T)*sum(psiVAlpha0s), ...
  + D*log(gamma(alpha0)), ...
  - D*sum(log(gamma(alpha))), ...
  + kappa1 * sumOfRhos, ...
  - sum(sum((model.vAlpha - 1) .* psiVAlpha)), ...
  + sum(psiVAlpha0s .* (vAlpha0s - T)), ...
  + sum(log(gammaVAlpha0s)), ...
  - sum(sum(log(gammaVAlpha))) ];

likeTerms = [ ... 
  sumOverTD(term1Fn), sumOverTD(term2Fn), ...
  sumOverD(term3Fn), sumOverTD(term4Fn), ...
  sumOverD(term5Fn), ...
  sumOverTD(term6Fn), sumOverTD(term7Fn), ...
  sumOverD(term8Fn), sumOverTD(term9Fn) ];

  
%cfLikeTerms - likeTerms

function sum = sumOverTD(fn);
global T;
global D;
sum = 0;
for t = 1:T
  for d = 1:D
    sum = sum + fn(t, d);
  end
end

function sum = sumOverD(fn);
global D;
sum = 0;
t = 0;
for d = 1:D
  sum = sum + fn(t, d);
end

