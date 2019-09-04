% function model = VEMInit(dataFile, T);
function model = VEMInit(dataFile, T);

% Load data
%data = hdf5load(dataFile);
%v = data.vs';
v = spconvert(load(dataFile));
[V, D] = size(v);

% Parameters governing initialization
% vMu initialization
MeanForVMu = ones(V, 1) / norm(ones(V, 1));
KappaForVMu = 5;
% alpha and vAlpha initialization
AlphaScale = 1;
vAlphaScale = 1;


% Variational parameters
alpha = ones(T, 1) * AlphaScale + 1;
M = ones(V, 1) / norm(ones(V, 1)); % Paramter to p(mu)
kappa0 = 10;
kappa1 = 5000;
xi = 5000;  % Scalar xi

vM = NormalizeColumns(rand(V, 1)); % Parameter to q(mu)
vMu = SampleVmf(KappaForVMu, MeanForVMu, T)'; % Parameter to q(phi)

% Initialize vAlpha
%vAlpha = rand(T, D) * vAlphaScale + 1;  % \tilde{\alpha}_d = vAlpha(:, d);
for d = 1:D
  distancesFromTopics = abs(CosineSimilarity(v(:, d), vMu)) + 0.01;
  vAlpha(:, d) = distancesFromTopics / sum(distancesFromTopics) * 3;
end


%vMu = max(vMu, 0.0001);
vMu = NormalizeColumns(vMu);

model = struct();
model.v = v;
model.V = V;
model.D = D;
model.T = T;
model.alpha = alpha;
model.kappa0 = kappa0;
model.kappa1 = kappa1;
model.xi = xi;
model.M = M;
model.vM = vM;
model.vMu = vMu;
model.vAlpha = vAlpha;
