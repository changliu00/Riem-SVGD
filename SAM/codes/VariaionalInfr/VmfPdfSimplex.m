% Plot the un-normalized pdf of the truncated vMF projected onto the
% unit simplex
% NOTE!  mu must have L2 norm = 1!
function [X,Y,Z] = VmfPdfSimplex(kappa, mu);

[m,n] = size(mu);
if m == 1
  mu = mu';
end

% can't use points where any component is zero
[X,Y] = meshgrid(.005:.01:1, .005:.01:1);
dimX = size(X, 1);
dimY = size(Y, 1);
Z = zeros(dimX, dimY);
for i = 1:dimX
  for j = 1:dimY
    x3 = 1 - X(i,j) - Y(i,j);
    if x3 <= 0 
      continue;
    end
    x1 = X(i,j);
    x2 = Y(i,j);
    
    %Z(i,j) = exp(kappa * (mu(1)*x1^0.5 + mu(2)*x2^0.5 + mu(3)*x3^0.5)) / ...
    %           (x1*x2)^0.5;
    
    Z(i,j) = VmfPdf(kappa, mu, [x1^0.5, x2^0.5, x3^0.5]) / (x1*x2)^0.5 / 4;
    %Z(i,j) = multipdf([x1, x2, x3], n, p);
  end
end
