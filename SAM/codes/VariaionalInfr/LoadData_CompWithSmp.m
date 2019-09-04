function [model, labels, J] = LoadData_CompWithSmp(model, dataFilename)

fp = fopen(dataFilename, 'r');
if(fp == -1)
	error('Cannot open data file!');
end
V = fscanf(fp, '%d', 1);
if isfield(model, 'V') && model.V ~= V
	fclose(fp);
	error('the V of model not match the V of data!');
end
model.V = V;
D = fscanf(fp, '%d', 1);
model.D = D;
model.v = zeros(V, D);
labels = zeros(1, D);
for i = 1:D
    numTerms = fscanf(fp, '%d', 1);
    labels(i) = fscanf(fp, '%d', 1);
	for j = 1:numTerms
        termID = fscanf(fp, '%d:', 1);
        value = fscanf(fp, '%f', 1);
		model.v(termID+1, i) = value;
	end
end
J = fscanf(fp, '%d', 1);
fclose(fp);

