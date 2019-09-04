% function result = rho(model, d);
function result = rho(model, d);

vAlphaD = model.vAlpha(:, d);
vAlphaD0 = sum(vAlphaD);
%result = dot(model.vMu * vAlphaD, model.v(:, d)) * (AvK(model.V, model.xi) / vAlphaD0 / sqrt(ESquaredNorm(model, d)));
result = vAlphaD' * (model.vMu' * model.v(:, d)) * (AvK(model.V, model.xi) / vAlphaD0 / sqrt(ESquaredNorm(model, d)));