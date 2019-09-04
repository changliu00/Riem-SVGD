% function grad = GradRhoVMu_BatchT(model, d);
%
% Just like GradRhoVMu, except it computes the gradient with respect to each
% topic for the specified document.  This is significantly more efficient than 
% calling GradRhoVMu for each topic individually.  Returns a V-by-T matrix where
% column t contains the gradient of rho(d) with respect to vMu_t.
function grad = GradRhoVMu_BatchT(model, d);

T = model.T;
D = model.D;
V = model.V;
xi = model.xi;

vAlphaD = model.vAlpha(1:T, d);
vAlphaD0 = sum(vAlphaD);
vd = model.v(1:V, d);
aXi = AvK(V, xi);

esn = ESquaredNorm(model, d);
gradEsns = zeros(V, T);
for t = 1:T
  gradEsns(:, t) = GradESquaredNorm(model, d, t);
end
    
%grad = aXi/vAlphaD0 * (vAlphaD(t)*vd/sqrt(esn) - dot(model.vMu*vAlphaD, vd)/(2*esn^(3/2))* gradEsn);
grad = aXi/vAlphaD0 * (1/sqrt(esn)*vd*vAlphaD' - dot(model.vMu*vAlphaD, vd)/(2*esn^(3/2))* gradEsns);