function [O,P_hat]= learn_P_hat(C, S_f, K, n_iter)
% Not used for the paper so far.
%
% It minimizes the function norm(C - HO, 'fro')^2 with respect to P_hat, O
% through an alternating minimization approach, with each step having a
% closed form solution. The matrix H is a type-II NV-GF.
%
%
%INPUT
%   - C: (factor of the) covariance matrix N x N
%   - S_f: dual GSO
%   - K: length of the filter
%   - useInput: boolean indicating whether input X should be used or not
%OUTPUT
%   - P_hat: N x K filter taps frequency domain
%   - O: orthogonal matrix

if nargin <4, n_iter=1; end


[N,T]= size(C);
O= RandOrthMat(N, 1e-9); % initialization unitary matrix
B= nan(N*T,N*K);
I= eye(N);
for n=1:n_iter
    
    %% P-Step: Learn P_hat for fixed orthogonal matrix O
    S_acc= I;
    B(:, 1:N)= kr(O', I);
    for k=1:K-1
        S_acc= S_f*S_acc;
        B(:, k*N+1: (k+1)*N)= kr(O', S_acc);
    end
    
    P_hat= reshape(B\C(:), [N K]);
    
    %% Orthogonal Procrustes:  Learn O with P_hat fixed
    
    H = NV_GF(P_hat, S_f, 'type-II');
    [U,~,V] = svd(C'*H);
    O = V*U';
    sprintf("Iteration %d: Objective value = %0.2f", n, norm(C - H*O, 'fro')^2)
    
end
end