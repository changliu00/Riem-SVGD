% function badWords = FindStopAndRareWords(v, stopWordThresh, rareWordThresh);
function badWords = FindStopAndRareWords(v, stopWordThresh, rareWordThresh);

proportionOfDocsContainingWord = sum(v ~= 0, 2) / size(v, 2);
badWords = (proportionOfDocsContainingWord > stopWordThresh) | ...
           (proportionOfDocsContainingWord < rareWordThresh);
