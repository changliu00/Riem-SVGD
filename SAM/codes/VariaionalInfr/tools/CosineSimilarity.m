% function similarities = CosineSimilarity(A, B);
%
% Computes the cosine similarity of the columns of A with the columns of B.
% Returns a matrix X such that Xij is the cosine similarity of A_i with B_j.
% In the case that norm(A_i) = 0 or norm(B_j) = 0, this
% implementation will return X_ij = 0.  If norm(A_i) = 0 AND norm(B_i) = 0,
% then X_ii = 0 as well.
function similarities = CosineSimilarity(A, B);

numColsA = size(A, 2);
numColsB = size(B, 2);

similarities = zeros(numColsA, numColsB);

% Normalize columns of A and B
aSquaredNorms = max(sum(A.^2, 1), 100*eps);
bSquaredNorms = max(sum(B.^2, 1), 100*eps);

A = A * diag(1./sqrt(aSquaredNorms)); 
B = B * diag(1./sqrt(bSquaredNorms));

similarities = A'*B;