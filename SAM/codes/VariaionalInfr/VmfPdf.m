% Computes densities for points X on the unit hypersphere under the von
% Mises-Fisher distribution. 
%
% Params:
%   k: kappa, the pdf concentration parameter
%   mu: the mean direction parameter; must be a vector with unit L2 norm
%   x: the points at which to evaluate the pdf. Each row should be a vector
%     with unit L2 norm
function densities = VmfPdf(k, mu, x);
n = length(mu); 
cpk = vmf_c(n, k);
for i = 1:size(x, 1)
  densities(i) = exp(k * dot(mu, x(i, :))) / cpk;
end

% Normalization constant for the von Mises-Fisher pdf
function c = vmf_c(n, k);
c = (2*pi)^(n/2) * besseli(n/2-1, k) / k^(n/2-1);