% function c = LogVmfC(n, k);
% Returns the log of the normalization constant for a Vmf distribution with
% dimensionality n and concentration parameter k.
function logc = LogVmfC(n, k);
% log of c = k^(n/2-1) / ( (2*pi)^(n/2) * besseli(n/2-1, k));
logc = (n/2-1)*log(k) - (n/2)*log(2*pi) - log(besseli(n/2-1, k)+eps);
