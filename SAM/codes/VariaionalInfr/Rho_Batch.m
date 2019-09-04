% function result = Rho_Batch(model);
function result = Rho_Batch(model);

% result = vAlphaD' * (model.vMu' * model.v(:, d)) * (AvK(model.V, model.xi) / vAlphaD0 / sqrt(ESquaredNorm(model, d)));

esns = ESquaredNorm_Batch(model);
vAlpha0s = sum(model.vAlpha, 1);

result = sum((model.vAlpha * SparseDiag(1./vAlpha0s./sqrt(esns))) .* (model.vMu' * model.v), 1) * AvK(model.V, model.xi);