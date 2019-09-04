function [documents, dictionary] = LoadCorpus(dataFile, dictFile, numDocuments, stopWordThresh, rareWordThresh);

if nargin < 3
  numDocuments = [];
end
if nargin < 4
  stopWordThresh = [];
end
if nargin < 5
  rareWordThresh = [];
end
   
% Load data
dictionary = LoadDictionary(dictFile); 
documents = spconvert(load(dataFile));
[V, D] = size(documents);
%if D > V
%  fprintf('Warning, data file has D > V; transposing\n');
%  documents = documents';
%  [V, D] = size(documents);
%end

% Randomly select (approximately) numDocuments docs
if ~isempty(numDocuments)
  documentsToSelect = find([rand(1, D) <= numDocuments / D]);
  documents = documents(:, documentsToSelect);
  [V, D] = size(documents);
end

if ~isempty(stopWordThresh) && ~isempty(rareWordThresh)
  [documents, dictionary] = RemoveBadWords(documents, dictionary, stopWordThresh, rareWordThresh);
end
documents = RemoveEmptyDocs(documents);
documents = NormalizeColumns(documents);


function docs = RemoveEmptyDocs(docs);
nonEmptyDocIndices = find(sum(docs ~= 0, 1) ~= 0);
docs = docs(:, nonEmptyDocIndices);
return;

function [docs, dict] = RemoveBadWords(docs, dict, stopWordThresh, rareWordThresh);
badWords = FindStopAndRareWords(docs, stopWordThresh, rareWordThresh);
numBadWords = full(sum(badWords));
fprintf('Throwing out %d stop/rare words\n', numBadWords);
goodWordIndices = find(~badWords);
docs = docs(goodWordIndices, :);
dict = dict(goodWordIndices);
return;

