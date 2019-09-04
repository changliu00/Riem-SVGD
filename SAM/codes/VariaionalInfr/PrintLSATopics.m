function PrintLSATopics(dataFile, dictFile, T, numTopWords, numBottomWords, fileHandle);

if nargin < 4
  numTopWords = 10;
end
if nargin < 5
  numBottomWords = 10;
end
if nargin < 6
  fileHandle = 1; % Standard output
end

[docs, dict] = LoadCorpus(dataFile, dictFile);
[V, D] = size(docs);

% Compute tf-idf; L1-normalize the documents
tf = docs * diag(1./sum(docs, 1));
idf = log(D ./ sum(docs ~= 0, 2));
tfIdf = tf .* repmat(idf, 1, D);

[U,D,V] = svds(tfIdf, T);

for t = 1:T
  [sortedWordWeights, sortedWordIndices] = sort(U(:, t), 'descend');
  topWordIndices = sortedWordIndices(1:numTopWords);
  topWordWeights = sortedWordWeights(1:numTopWords);
  topWords = dict(topWordIndices);
  
  bottomWordIndices = sortedWordIndices(end:-1:(end-numBottomWords+1));
  bottomWordWeights = sortedWordWeights(end:-1:(end-numBottomWords+1));
  bottomWords = dict(bottomWordIndices);

  fprintf(fileHandle, 'LSA Topic %d\n', t);
  fprintf(fileHandle, '--------\n');
  fprintf(fileHandle, '  Highest-weighted words:\n');
  for w = 1:numTopWords
    fprintf(fileHandle, '    %.4f  %s\n', topWordWeights(w), topWords{w});
  end
  fprintf(fileHandle, '\n');
  
  if numBottomWords ~= 0
    fprintf(fileHandle, '  Lowest-weighted words:\n');
    for w = 1:numBottomWords    
      fprintf(fileHandle,'    %.4f  %s\n', bottomWordWeights(w), bottomWords{w});
    end
    fprintf(fileHandle, '\n');
  end
end
