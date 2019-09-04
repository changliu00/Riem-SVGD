function data = NormalizeColumns(data, normType);

% It's not possible to normalize a vector that has zero norm.  If a
% column's norm is less than THRESHOLD, we do not attempt to normalize it.
THRESHOLD = 100*eps;

if nargin < 2
  normType = 2;  % Euclidean norm
end

switch normType
  case 1  % Sum norm
    columnNorms = sum(data, 1);
    columnNorms = columnNorms .* (columnNorms > THRESHOLD) + (columnNorms <= THRESHOLD);
    normalizingMatrix = SparseDiag(1./columnNorms);
  case 2  % Euclidean norm
    columnSquaredNorms = sum(data.^2, 1);
    columnSquaredNorms = columnSquaredNorms .* (columnSquaredNorms > THRESHOLD) + ...
                         (columnSquaredNorms <= THRESHOLD);
    normalizingMatrix = SparseDiag(1./sqrt(columnSquaredNorms));
  case 'inf'  % Infinity norm
    columnNorms = max(data, [], 1);
    columnNorms = columnNorms .* (columnNorms > THRESHOLD) + ...
                         (columnNorms <= THRESHOLD);
    normalizingMatrix = SparseDiag(1./columnNorms);
  otherwise
    error(['Invalid norm ' normType]);
end

data = data * normalizingMatrix;
