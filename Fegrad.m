function G = Fegrad(data, sigma, x, W, lambda)
% See NB 37, Dec. 17, 2018

    N = size(data, 2);

    Tx = T(data, sigma, x);
    
    G.x = real(N*x - ifft(sum(conj(fft(W)).*fft(data), 2)))/sigma^2;
    G.W = Tx + log(W) + 1;
    
    % Barrier
    if ~exist('lambda', 'var') || isempty(lambda)
        lambda = 0;
    end
    if lambda ~= 0
        G.W = G.W - lambda./W;
    end
    
    % Normalization
    G.x = G.x / numel(data);
    G.W = G.W / numel(data);

end

function Tmat = T(data, sigma, x)

    Tmat = (norm(x, 2).^2 + repmat(sum(data.^2, 1), length(x), 1) - 2*real(ifft(bsxfun(@times, fft(data), conj(fft(x))))))/(2*sigma^2);
    
end
