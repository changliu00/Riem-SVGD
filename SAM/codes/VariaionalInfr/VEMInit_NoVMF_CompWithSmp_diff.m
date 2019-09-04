% function model = VEMInit_NoVMF(documentMatrix, T);
function model = VEMInit_NoVMF_CompWithSmp_diff(model, T, max_iter)

D = model.D;
V = model.V;

M = sum(model.v, 2);
M = M / norm(M, 2);
% % check:
% norm(M, 2)
% sum(v .* v, 1)
% pause;

% Parameters governing initialization
% vMu initialization
% MeanForVMu = ones(V, 1) / norm(ones(V, 1));
% KappaForVMu = 5;
% alpha and vAlpha initialization
xi = 1e4;  % Scalar xi
kappa0 = 1e4;
kappa1 = 3e4;
AlphaScale = 10;
% vAlphaScale = 10;

% Variational parameters
alpha = ones(T, 1) * AlphaScale;
vAlpha = zeros(T, D);
% alpha = ones(T, 1) * AlphaScale + 1;
% M = ones(V, 1) / norm(ones(V, 1)); % Paramter to p(mu)

vM = full(NormalizeColumns(rand(V, 1))); % Parameter to q(mu)
vMu = full(NormalizeColumns(rand(V, T))); % Parameter to q(phi)

% Initialize vAlpha
%vAlpha = rand(T, D) * vAlphaScale + 1;  % \tilde{\alpha}_d = vAlpha(:, d);
for d = 1:D
  distancesFromTopics = abs(CosineSimilarity(model.v(:, d), vMu)) + 0.01;
  vAlpha(:, d) = distancesFromTopics / sum(distancesFromTopics) * 3;
end

% model = struct(); % already there at the beginning
model.T = T;
model.alpha = alpha;
model.kappa0 = kappa0;
model.kappa1 = kappa1;
model.xi = xi;
model.M = M;
model.vM = vM;
model.vMu = vMu;
model.vAlpha = vAlpha;

model.elapsedTime = 0;
model.iteration = 0;
model.max_iter = max_iter;

