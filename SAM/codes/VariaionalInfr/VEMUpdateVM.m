function model = VEMUpdateVM(model);
% Old: multiple topic means; vM is a V-by-T matrix
%T = model.T;
%model.vM = model.kappa0*(model.M * ones(1, T)) ...
%           + AvK(model.V, model.xi)*model.xi*model.vMu;
%model.vM = NormalizeColumns(model.vM);

model.vM = model.kappa0*model.M ...
           + AvK(model.V, model.xi)*model.xi*sum(model.vMu, 2);
model.vM = NormalizeColumns(model.vM);