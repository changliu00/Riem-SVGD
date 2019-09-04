% function PrintTopics(model, dict, numTopWords, numBottomWords[, fileHandle])
%
% Prints the specified number of highest- and lowest-weighted words from
% each of the model's topics.  An open file handle may be given in the 
% fileHandle argument in order to print the topics to a text file.  By
% default, the topics are printed to standard output.
function PrintTopics(model, dict, numTopWords, numBottomWords, fileHandle);

if nargin < 5
  fileHandle = 1; % Standard output
end

[y, i] = sort(model.vMu, 1, 'descend');

for t = 1:model.T
  topWordIndices = i(1:numTopWords, t);
  topWords = dict(topWordIndices);
  topWordWeights = model.vMu(topWordIndices, t); 
  
  bottomWordIndices = i(end:-1:(end-numTopWords+1), t);
  bottomWords = dict(bottomWordIndices);
  bottomWordWeights = model.vMu(bottomWordIndices, t);
  
  fprintf(fileHandle, 'Topic %d\n', t);
  fprintf(fileHandle, '--------\n');
  fprintf(fileHandle, '  Highest-weighted words:\n');
  for w = 1:numTopWords
    fprintf(fileHandle, '    %.4f  %s\n', topWordWeights(w), topWords{w});
  end
  fprintf(fileHandle, '\n');
  fprintf(fileHandle, '  Lowest-weighted words:\n');
  for w = 1:numBottomWords    
    fprintf(fileHandle,'    %.4f  %s\n', bottomWordWeights(w), bottomWords{w});
  end
  
  fprintf(fileHandle, '\n');
  fprintf(fileHandle, '\n');
end