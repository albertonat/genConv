function r = stationarity_coefficient(C)
%STATIONARITY_COEFFICIENT It computes the stationarity coefficient of a
%random graph process Y with spectral covariance matrix C
%   
%
% C is usually the covariance matrix in the Fourier domain, i.e.,
%   C = V'C_y V, with V the eigenvectors of the GSO S. If the process Y is
%   stationary on S, the matrix C should be as much diagonal as possible.
%   As such, the stationarity coefficient is computed as the ratio between
%   the energy of the diagonal and the energy of the matrix itself.
%

    r = norm(diag(C))/norm(C, 'fro');

end

