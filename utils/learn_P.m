function P= learn_P(X,Y,S,L, useInput)
%
%INPUT
%   - X: input data N x T
%   - Y: output data N x T
%   - S: primal GSO
%   - L: length of the filter
%   - useInput: boolean indicating whether input X should be used or not
%OUTPUT
%   - Y: output data N x T

if ~exist('useInput', 'var'), useInput= true; end

%% LEAST SQUARES WITH INPUT-OUTPUT
if useInput % Learn filter taps with Input-Output (X,Y)
    debug=0;
    [N,T]= size(X);
    A= nan(N*T,N*L);
    I= eye(N);
    X_acc= X;
    A(:, 1:N)= kr(X', I);
    for l=1:L-1
        X_acc= S*X_acc;
        A(:, l*N+1: (l+1)*N)= kr(X_acc', I);
        if debug
            sprintf("l=%d \n Rank X_acc: %d \n Rank kr(X_acc', I): %d \n Rank of filled A: %d", ...
                l, rank(X_acc), rank(kr(X_acc', I)),rank(A(:,1: (l+1)*N)))
        end
    end
    
    P= reshape(A\Y(:), [N L]);
    
    
    %% Blind Method: AM over P and unitary U
else % Learn filter taps with Output-only Y BLIND METHOD
    [N, T] = size(Y);
    Ry = (1/T)*(Y*Y'); % empirical covariance matrix
    [U_y,Lambda_y]= eig(Ry); %factorization of Ry
    R= U_y*sqrt(Lambda_y); % first term objective
    U= RandOrthMat(N, 1e-9); % initialization unitary matrix
    niter=1000;
    P= nan(N,L);
    for i=1:niter
        %--- P-step -----
        A= nan(N*N,N*L);
        I= eye(N);
        
        X_acc= U;
        A(:, 1:N)= kr(U', I); % Khatri-Rao - columnwise Kronecker
        for l=1:L-1
            X_acc= S*X_acc;
            A(:, l*N+1: (l+1)*N)= kr(X_acc', I);
        end
        
        P= reshape(A\R(:), [N L]);
        
        
        %---- U-step ------
        H_I= diag(P(:,1));
        S_tmp=speye(N);
        for l=2:L
            S_tmp= S_tmp*S;
            H_I= H_I + diag(P(:,l))*S_tmp;
        end
        
        [W, ~, Z]= svd(R.'*H_I);
        U= Z*W';
        
        sprintf("Iteration %d: Objective value = %0.2f", i, norm(R - H_I*U, 'fro')^2)
    end % end niter A-M
    
end

end