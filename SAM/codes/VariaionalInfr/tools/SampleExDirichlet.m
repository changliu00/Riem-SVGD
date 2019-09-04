% r = SampleExDirichlet(alpha, d, n);
% Samples n (d-1)-dimensional vectors parameters from 
% Dirichlet(alpha, alpha, ..., alpha), i.e., the exchangable Dirichlet.
%
% Adapted from:
%   http://www.mathworks.com/matlabcentral/newsreader/view_thread/240193
function r = SampleExDirichlet(alpha, d, n);

r = gamrnd(alpha, 1, n, d);  % returns n-by-d array of gamma(alpha, 1) samples
r = r(:,1:end-1) ./ repmat(sum(r,2),1,d-1);  
