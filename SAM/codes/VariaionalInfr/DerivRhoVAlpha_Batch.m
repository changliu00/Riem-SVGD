% function derivs = DerivRhoVAlpha_Batch(model);
% 
% This function does the same thing as DerivRhoVAlpha, except it computes the 
% derivatives with respect to all vAlphas at once.  This removes a lot of
% redundant computation and is significantly faster than calling DerivRhoVAlpha
% over and over again.
function derivs = DerivRhoVAlpha_Batch(model);

T = model.T;
xi = model.xi;
V = model.V;
D = model.D;

%vAlphaD = model.vAlpha(1:T, d);
vAlphaD0s = sum(model.vAlpha, 1);

aXi = AvK(V,xi);

% A T-by-D matrix containing the derivative of ESN w.r.t. each vAlpha_t,d
derivsOfESquaredNorm = DerivESquaredNorm_Batch(model);

% A 1-by-D vector containing the expected squared norms of vMu*vAlpha(:, d)
esns = ESquaredNorm_Batch(model);

vMuDotVd = model.vMu' * model.v;  % T-by-D

%vMuTimesVAlpha = model.vMu * model.vAlpha; % One column for each document
%vMuTimesVAlphaDotVd = sum(vMuTimesVAlpha .* model.v, 1); % 1-by-D
% Same thing, but way faster and not requiring V*D memory
vMuTimesVAlphaDotVd = sum(model.vAlpha .* vMuDotVd, 1);


% derivs = zeros(T, D);
% for d = 1:D
%   for t = 1:T
%     firstTerm = (vMuDotVd(t, d)/vAlphaD0s(d) - vMuTimesVAlphaDotVd(d)/vAlphaD0s(d)^2) / sqrt(esns(d));
%     secondTerm = -vMuTimesVAlphaDotVd(d)/vAlphaD0s(d) / (2*esns(d)^(3/2)) * derivsOfESquaredNorm(t, d);
%     derivs(t, d) = aXi * (firstTerm + secondTerm);
% 
%     %firstTerm = (model.vMu(1:V, t)/vAlphaD0s(d) - vMuTimesVAlpha(1:V, d)/vAlphaD0s(d)^2) / sqrt(esns(d));
%     %secondTerm = -vMuTimesVAlpha(1:V, d)/vAlphaD0s(d) / (2*esns(d)^(3/2)) * derivsOfESquaredNorm(t, d);
%     %derivs(t, d) = aXi * dot((firstTerm + secondTerm), model.v(:, d));
%   end
% end

% Same thing as the nested loop above, except way faster

derivs = vMuDotVd * SparseDiag(1./vAlphaD0s);

% Subtract a constant from each column
derivs = bsxfun(@minus, derivs, vMuTimesVAlphaDotVd ./ vAlphaD0s.^2);

% Divide each column by sqrt(esns(d))
derivs = derivs * SparseDiag(1./sqrt(esns));

s = vMuTimesVAlphaDotVd ./ vAlphaD0s ./ (2*esns.^(3/2));
derivs = aXi * (derivs - derivsOfESquaredNorm * SparseDiag(s));

