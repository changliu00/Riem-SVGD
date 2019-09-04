% function likelihood = L_kappa1(model);
function likelihood = L_kappa1(model);
 
D = model.D;
V = model.V;
k1 = model.kappa1;

sumOfRhos = sum(Rho_Batch(model));
likelihood = D * LogVmfC(V,k1) + k1*sumOfRhos;
