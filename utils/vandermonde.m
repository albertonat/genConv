function A = vandermonde(v,k)
%VANDERMONDE Vandermonde matrix up to the (k-1) th power.
%   A = vander(v,k), for a vector of length n, returns the n-by-k
%   Vandermonde matrix A. The columns of A are powers of the vector v,
%   such that the k-th column is A(:,k) = v.^(k-1).
%   Notice that vandermonde(v,1) = v in our implementation.
%
%   Class support for input v:
%      float: double, single

arguments
    v
    k = length(v) % deafult value
end

A = repmat(v, 1, k);
A(:, 1) = 1;
A = cumprod(A, 2);

end