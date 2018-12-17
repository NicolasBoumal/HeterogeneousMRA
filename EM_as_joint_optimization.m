% See NB 37, Dec. 17, 2018

%% Problem setup

% signal length
L = 10;
% Ground truth signal
x_true = randn(L, 1);
% Noise level
sigma = 3;  % multinomial will cause trouble if we converge to a W with zeros in it -- if sigma is large, that's less likely to happen.

% Generate data
number_of_measurements = 1e5;
data = generate_observations(x_true, number_of_measurements, sigma);

%% 
elements.x = euclideanfactory(L, 1);
elements.W = multinomialfactory(L, number_of_measurements);
problem.M = productmanifold(elements);
problem.cost = @(pair) Fcost(data, sigma, pair.x, pair.W);
problem.egrad = @(pair) Fegrad(data, sigma, pair.x, pair.W);

% checkgradient(problem);
% return;

pair = conjugategradient(problem);
x_est = pair.x;
W_est = pair.W;

fprintf('Relative error full optimization: %g\n', relative_error(x_true, x_est));

x_em = MRA_EM(data, sigma);
fprintf('Relative error traditional EM:    %g\n', relative_error(x_true, x_em));
