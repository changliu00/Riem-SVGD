function model = VEMUpdateM(model);

% Update M
model.M = sum(model.vM, 2); % Sum across topics
model.M = model.M / norm(model.M);
