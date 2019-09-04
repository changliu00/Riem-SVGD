% function dict = LoadDictionary(dictFile);
% Returns a dictionary as a cell array of strings
function dict = LoadDictionary(dictFile);

f = fopen(dictFile);
dict = cell(0);
while 1
  line = fgetl(f);
  if ~ischar(line)
    break
  end
  dict{end+1} = line;
end
fclose(f);