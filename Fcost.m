function f = Fcost(data, sigma, x, W)
% See NB 37, Dec. 17, 2018

    Tx = T(data, sigma, x);
    
%     %-- check
%     D = zeros(size(data));
%     for i = 1 : size(data, 1)
%         for j = 1 : size(data, 2)
%             D(i, j) = Tx(i, j) - norm(circshift(x, i-1) - data(:, j))^2 / (2*sigma^2);
%         end
%     end
%     fprintf('-- check: %g --\n', norm(D, 'fro')); % should print 0 or close
%     %-- endcheck
    
    f = inner(W, Tx + log(W));

end

function Tmat = T(data, sigma, x)

    % Much can be saved here: fft(data) should be saved; there's some issue
    % with computing the difference of large similar numbers, ...
    Tmat = (norm(x, 2).^2 + repmat(sum(data.^2, 1), length(x), 1) - 2*real(ifft(bsxfun(@times, fft(data), conj(fft(x))))))/(2*sigma^2);

    % For real x, Tmat(i, j) = norm(circshift(x, i-1) - data(:, j))^2 / (2*sigma^2)
    
end

function ip = inner(A, B)
    ip = real(A(:)'*B(:));
end
