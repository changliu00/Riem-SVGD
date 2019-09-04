% function avk = AvK(v,k);
function avk = AvK(v,k);
% Bad for large V ?
avk = ((sqrt((v/k)^2+4) - v/k)/2);
