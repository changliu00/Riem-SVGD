% function grad = GradRhoVMu(model, d, t);
function grad = GradRhoVMu(model, d, t);

T = model.T;
D = model.D;
V = model.V;
xi = model.xi;

vAlphaD = model.vAlpha(1:T, d);
vAlphaD0 = sum(vAlphaD);
vd = model.v(1:V, d);
aXi = AvK(V, xi);

esn = ESquaredNorm(model, d);
gradEsn = GradESquaredNorm(model, d, t);
    
grad = aXi/vAlphaD0 * (vAlphaD(t)*vd / sqrt(esn) - dot(model.vMu*vAlphaD, vd)/(2*esn^(3/2)) * gradEsn);