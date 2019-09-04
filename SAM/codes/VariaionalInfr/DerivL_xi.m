% function deriv = DerivL_xi(model);
function deriv = DerivL_xi(model);
V = model.V;
T = model.T;
D = model.D;
xi = model.xi;
k0 = model.kappa0;
k1 = model.kappa1;
aXi = AvK(model.V, model.xi);
aPrimeXi = DerivAvK(model.V, model.xi);
aK0 = AvK(model.V, model.kappa0);

% sumOverDocuments = 0;
% for d = 1:D
%   sumOverDocuments = sumOverDocuments + DerivRhoXi(model, d);
% end

sumOverDocuments = sum(DerivRhoXi(model));

% Old: vM is a V-by-T matrix
%deriv = (aPrimeXi*xi + aXi) * (aK0*sum(sum(model.vM .* model.vMu)) - T) ...
%        + k1*sumOverDocuments;
      
deriv = (aPrimeXi*xi + aXi) * (aK0*(model.vM'*sum(model.vMu, 2)) - T) ...
        + k1*sumOverDocuments;