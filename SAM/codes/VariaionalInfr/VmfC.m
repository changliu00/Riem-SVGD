% function c = VmfC(n, k);
% Returns the normalization constant for a Vmf distribution with
% dimensionality n and concentration parameter k.
function c = VmfC(n, k);
c = k^(n/2-1) / ( (2*pi)^(n/2) * besseli(n/2-1, k) );