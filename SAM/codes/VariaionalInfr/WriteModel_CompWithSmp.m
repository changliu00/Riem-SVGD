function WriteModel_CompWithSmp(model, filename)
%

K = model.T;
V = model.V;

fp = fopen(filename, 'w');
fprintf(fp, 'vem: %d\n', model.iteration);
% basic parameters
fprintf(fp, '%.3f %d %d %d %d %d %d %d %d %d %.6e\n', model.elapsedTime, V, model.D, K, 0, 0, 0, 0, 0, 0, 0);
% model parameters
fprintf(fp, '%.6e %.6e %.6e\n', model.xi, model.kappa0, model.kappa1);
% alpha
fprintf(fp, ' %.6e', model.alpha); fprintf(fp, '\n');
% M
fprintf(fp, ' %.6e', full(model.M)); fprintf(fp, '\n');
% beta
for i = 1:K
	fprintf(fp, ' %.6e', model.vMu(:, i)); fprintf(fp, '\n');
end
fclose(fp);

