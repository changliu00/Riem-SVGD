% samples = SampleVmf(k, mu, numSamples);
function samples = SampleVmf(k, mu, numSamples);

if nargin < 3
  numSamples = 1;
end

% Make mu a column vector
if size(mu, 1) == 1
  mu = mu';
end

% Initialization
d = length(mu);
t1 = sqrt(4*k*k+(d-1)*(d-1));
b = (-2*k+t1)/(d-1);
x0=(1-b)/(1+b);
m=(d-1)/2.0;
c=k*x0+(d-1)*log(1-x0*x0);

% generate a matrix that rotates [0,0,..,1] onto mu
% (the samples will initially be around [0,0,..,1])
if d==1
  error('DIM needs to be at least 2');
end

n = length(mu);
i = zeros(n, 1);
i(n) = 1;
rm = gen_rotmat(i, mu);  % TODO

            
% Generate the samples
for index = 1:numSamples
  s = zeros(d, 1);
  t = -1000;
  u = 1.0;
  dm = d-1;
  while (t<log(u))
    % sample from beta
    z = betarnd(m, m);
    u = rand(1);
    w = (1-(1+b)*z)/(1-(1-b)*z);
    % Next line is wrong in Dhillon & Sra
    t = k*w+dm*log(1-x0*w)-c;
  end
  
  % generate random unit vector on d-1 dimensional sphere
  v = sph_rand(dm); % TODO
  s(1:dm) = sqrt(1-w*w)*v;
  s(end) = w;
  % rotate to mu
  s = rm * s;
  
  samples(index, 1:d) = s;
end



function rot = gen_rotmat(p,q);
q = q/norm(q);
p = p/norm(p);
npq = norm(p-q);
if npq < 1e-5
  n = length(p);
  rot = eye(n);
else
  rot = gen_refmat(q, -p) * gen_refmat(p, -p);
end
  
  
function ref = gen_refmat(p, q);
%    Return a (left multiplying) matrix that reflects p onto q.
%    p and q can be of arbitrary dimension.

%    @param p,q: vectors, same length
%    @type p,q: numpy array (shape=n, with n>1)

%    @return: matrix that reflects p onto q
%    @rtype: numpy array (shape=(n,n))
q = q/norm(q);
p = p/norm(p);
pq = p-q;
npq = norm(pq);
if npq < 1e-5
  n = length(p);
  ref = eye(n);
else
  pq = pq/npq;
  n = length(p);
  %pq.shape=(n,1)  
  % This makes sure it acts like a column vector, for the outer
  % product below
  i = eye(n);
  ref = eye(n) - 2*pq*pq';
end

function a = sph_rand(n);
% Return a random vector on the n-dimensional sphere S.
% @param n: dimension of the sphere
% @type n: int
% @return: random vector on the n-dimensional sphere
% @rtype: numpy array, shape=3, type='d'
% Simply generate n gaussian numbers and normalize
% From Knuth's book
a = randn(n, 1);
a = a/norm(a);
