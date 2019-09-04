% function m = SparseDiag(diagonalElements);
function m = SparseDiag(diagonalElements);

[a, b] = size(diagonalElements);
assert(a == 1 || b == 1, 'Argument must be a vector');

if a == 1
  diagonalElements = diagonalElements';
end

len = length(diagonalElements);
m = spdiags(diagonalElements, 0, len, len);