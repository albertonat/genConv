function C = learn_C(Cy,S_f, Psi, K, L)
% Introduced section. To check.
%
%INPUT
%   - Cy: square root covariance matrix N x N
%   - S_f: dual GSO
%   - Psi: Vandermonde matrix primal eigenvalues N x L
%   - L: length of the filter primal
%   - K: length of the filter dual domain
%OUTPUT
%   - C= filter expansion coefficients
%% LEAST SQUARES WITH INPUT-OUTPUT
    N= size(Cy,1);
    B= nan(N*N,N*K);
    I= eye(N);
    %O= RandOrthMat(N);
    O=eye(N); % in principle every orthogonal matrix is ok
    S_acc= I;
    B(:, 1:N)= kr(I, I);
    for k=1:K-1
        S_acc= S_f*S_acc;
        B(:, k*N+1: (k+1)*N)= kr(O, S_acc);
    end
    blk= repmat({Psi}, 1, K);
    B= B*blkdiag(blk{:});
    C_T= reshape(B\Cy(:), [L K]);
    C= C_T';

end