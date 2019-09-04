% function deriv = DerivL_kappa0(model);
function deriv = DerivL_kappa0(model);

k0 = model.kappa0;
V = model.V;
T = model.T;
xi = model.xi;

aXi = AvK(V,xi);
aK0 = AvK(V,k0);
aPrimeK0 = DerivAvK(V,k0);

% %sumOverTopics = sum(sum(model.vM .* (model.M*ones(1,T) + aXi*xi*model.vMu)));
% %sumOverTopics = dot(sum(model.vM, 2), model.M) ...
% %                + aXi*xi*sum(sum(model.vM .* model.vMu));
% 
% %deriv = aPrimeK0*(k0*sumOverTopics - T*k0) ...
% %        + aK0*(sumOverTopics - T);
% 
%       
% deriv = aPrimeK0 * ...
%           (dot(sum(model.vM, 2), k0*model.M) ...
%            + aXi*xi* sum(sum(model.vM .* model.vMu)) ...
%            - T*k0) ...
%          + aK0 * (dot(sum(model.vM, 2), model.M) - T);
%         
% %deriv = ...
% %  aXi * sum(model.M' * model.vM) ...
% %  + aPrimeK0 * (aXi * xi * sum(sum(model.vM .* model.vMu)) - T*k0) ...
% %  - T*aK0;



mDotVmMinusOne = model.M'*model.vM - 1;

deriv = aPrimeK0 * (k0*mDotVmMinusOne + aXi*xi*model.vM'*sum(model.vMu, 2)) ...
        + aK0 * mDotVmMinusOne;