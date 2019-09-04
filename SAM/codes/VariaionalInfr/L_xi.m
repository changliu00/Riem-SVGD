% function likelihood = L_xi(model);
function likelihood = L_xi(model);

V = model.V;
T = model.T;
D = model.D;
k0 = model.kappa0;
k1 = model.kappa1;
xi = model.xi;

% sumOfRhos = 0;
% for d = 1:D
%   sumOfRhos = sumOfRhos + rho(model, d);
% end

%likelihood = AvK(V,xi)*xi * (AvK(V,k0)*sum(sum(model.vM .* model.vMu)) - T) ...
%             + k1*sumOfRhos;

sumOfRhos = sum(Rho_Batch(model));
likelihood = AvK(V,xi)*xi * (AvK(V,k0)*model.vM'*sum(model.vMu, 2) - T) ...
             + k1*sumOfRhos;