% function likelihood = L_kappa0(model);
function likelihood = L_kappa0(model);

k0 = model.kappa0;
V = model.V;
T = model.T;
xi = model.xi;

aXi = AvK(V,xi);
aK0 = AvK(V,k0);

likelihood = aK0 * (k0*(model.vM' * model.M - 1) + aXi*xi*model.vM'*sum(model.vMu, 2));
                         