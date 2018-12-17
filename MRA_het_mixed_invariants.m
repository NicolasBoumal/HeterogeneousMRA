function [X_est, problem] = MRA_het_mixed_invariants(data, sigma, K, X0, opts, w, nextrainits)

    M = size(data, 2);
    
    if ~exist('w', 'var') || isempty(w)
        w = ones(M, 1);
    end
    assert(length(w) == M, 'w must have length M');
    assert(all(w >= 0), 'w must be nonnegative');
    w = w / sum(w);
    
    if ~exist('nextrainits', 'var') || isempty(nextrainits)
        nextrainits = 0;
    end

    [mumix, psmix, Bmix] = invariants_from_data_no_debias(data, w);
    
    [X_est, problem] = MRA_het_mixed_invariants_from_invariants_no_debias(mumix, psmix, Bmix, sigma, K, X0, opts, nextrainits);

end

