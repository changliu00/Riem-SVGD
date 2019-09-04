% r = SampleDirichlet(alpha, n);
% Samples n (d-1)-dimensional vectors parameters from 
%   Dirichlet(alpha_1, alpha_2, ..., alpha_d)
% as an n-by-(d-1) matrix.
% 
% Stolen from:
%   http://www.mathworks.com/matlabcentral/newsreader/view_thread/240193
function r = SampleExDirichlet(a, n);

% alpha must be a row vector
if size(a, 1) > size(a, 2)
  a = a';
end

% returns n-by-d array of gamma(alpha, 1) samples
r = gamrnd(repmat(a,n,1),1,n,length(a));
r = r(:,1:end-1) ./ repmat(sum(r,2),1,length(a)-1);