% function grad = GradESquaredNorm(model, d, t);
function grad = GradESquaredNorm(model, d, t);

T = model.T;
D = model.D;
V = model.V;
xi = model.xi;

vAlphaD = model.vAlpha(1:T, d);
vAlphaD0 = sum(vAlphaD);

aXiSquared = AvK(model.V, model.xi)^2;

grad = 1/(vAlphaD0*(vAlphaD0+1)) * ...
       (2*(1-aXiSquared)*model.vMu(1:V, t) + 2*aXiSquared*vAlphaD(t)*(model.vMu*vAlphaD));

