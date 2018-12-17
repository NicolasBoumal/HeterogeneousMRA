clear all; %#ok<CLALL>
close all;
clc;

%% Problem setup

% signal length
L = 20;
% Ground truth signal
x_true = randn(L, 1);
% Noise level
sigma = 3;

%%
clf;
subplot(1, 2, 1);

%% Data generation

number_of_measurements = 1e7;
data = generate_observations(x_true, number_of_measurements, sigma);

fprintf('About to run on %g measurements.\n', number_of_measurements);
pause(2);

%% Online EM trial
batch_size = 1000;
number_of_batches = floor(number_of_measurements/batch_size); % if division is not exact, we ignore the remaining incomplete batch.
erroem = zeros(number_of_batches, 1);
x0 = randn(size(x_true));
x = x0;
fftx = fft(x);
for k = 1 : number_of_batches
    Y = data(:, (k-1)*batch_size + (1:batch_size));
    fftY = fft(Y);
    C = ifft(bsxfun(@times, conj(fftx), fftY));
    sqnormY = repmat(sum(abs(Y).^2, 1), L, 1);
    T = (2*C - sqnormY)/(2*sigma^2);
    T = bsxfun(@minus, T, max(T, [], 1));
    W = exp(T);
    W = bsxfun(@times, W, 1./sum(W, 1));
    fftx_new = mean(conj(fft(W)).*fftY, 2);
    fftx = ((k-1)*fftx + fftx_new)/k; % online averaging: up to that point, fftx had been constructed based on (k-1)*batch_size data, and fftx_new is based on batch_size new data.
    erroem(k) = relative_error(x_true, ifft(fftx));
%     if mod(k, 1e3) == 0
%         loglog(batch_size*(1:number_of_batches), erroem); drawnow;
%     end
end
loglog(batch_size*(1:number_of_batches), erroem);
hold all;

xlabel('Number of observations seen');
ylabel('Relative error wrt ground truth');
title(sprintf('L = %d, \\sigma = %.2g', L, sigma));
set(gcf, 'Color', 'w');

%% Compare with online method of moments
batch_size = 1000; % contrary to Online EM, here, the batch size only affects our sampling of the error curve: it doesn't affect the actual errors.
number_of_batches = floor(number_of_measurements/batch_size); % if division is not exact, we ignore the remaining incomplete batch.
errmom = zeros(number_of_batches, 1);
x = x0;
mean_est = 0;
P_est = zeros(L, 1);
B_est = zeros(L, L);
for k = 1 : number_of_batches
    Y = data(:, (k-1)*batch_size + (1:batch_size));
    [mean_add, P_add, B_add] = invariants_from_data_no_debias(Y);
    % Online averaging of the moments
    mean_est = ((k-1)*mean_est + mean_add)/k;
    P_est    = ((k-1)*P_est    + P_add   )/k;
    B_est    = ((k-1)*B_est    + B_add   )/k;
    nextrainits = 0; % increase if optimization seems to land in bad local optima
    x = MRA_het_mixed_invariants_from_invariants_no_debias(mean_est, P_est, B_est, sigma, 1, x, [], nextrainits); % warm start
    errmom(k) = relative_error(x_true, x);
end
loglog(batch_size*(1:number_of_batches), errmom);

%% Compare with a few iterations of EM on all of the data
Y = generate_observations_het(x_true, number_of_measurements, sigma);
x = x0;
fftx = fft(x);
n_iter = 150;
errfem = zeros(n_iter, 1);
fftY = fft(data);
for iter = 1 : n_iter
    C = ifft(bsxfun(@times, conj(fftx), fftY));
    sqnormY = repmat(sum(abs(Y).^2, 1), L, 1);
    T = (2*C - sqnormY)/(2*sigma^2);
    T = bsxfun(@minus, T, max(T, [], 1));
    W = exp(T);
    W = bsxfun(@times, W, 1./sum(W, 1));
    fftx_new = mean(conj(fft(W)).*fftY, 2);
    fftx = fftx_new;
    errfem(iter) = relative_error(x_true, ifft(fftx));
end
for k = 1 : n_iter
    loglog([1, number_of_measurements], errfem(k)*[1, 1], 'k--');
    hold all;
end

subplot(1, 2, 2);
semilogy(1 : n_iter, errfem, 'k.-');
title('Relative error full EM iterations');
xlabel('iterations');

%%
savefig(gcf, 'online_em.fig');