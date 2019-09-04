% function deriv = DerivAvK(v,k)
% Derivative of the VMF mean resultant length w.r.t. kappa.
function deriv = DerivAvK(v,k)

% From Sra & Dhillon, TR-03-06
%a = AvK(v,k);
%deriv = 1-a^2 - (v-1)/k*a;

deriv = -1/2/(v^2/k^2+4)^(1/2)*v^2/k^3+1/2*v/k^2;