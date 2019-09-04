% Evaluates the portion of the log-likelihood lower bound that depends on
% \tilde{mu_t}
function likelihood = L_vMu(model);

% likelihood = 0;
% for t = 1:model.T
%   likelihood = likelihood + L_vMuT(model, t);
% end
% return;

D = model.D;
V = model.V;
xi = model.xi;
aXi = AvK(V, xi);
aK0 = AvK(V, model.kappa0);

sumOfRhos = sum(Rho_Batch(model));

vMDotSumOfVMu = model.vM' * sum(model.vMu, 2);
likelihood = aXi*aK0*xi*vMDotSumOfVMu + model.kappa1*sumOfRhos;
