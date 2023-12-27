function S = GSO(A, type)
%GSO Return the GSO S specified by type given the adjacency matrix A
%   It creates the GSO.

A= double(A);

switch type
    case 'laplacian'
        D= diag(A* ones(size(A,1),1));
        S= sparse(D-A);
    case 'adjacency'
        S= sparse(A);
    case 'normalized-adjacency'
        % Makes its maximum eigenvalue 1
        d= A* ones(size(A,1),1);
        S= sparse(diag(1./sqrt(d))*A*diag(1./sqrt(d)));
    case 'normalized-laplacian'
        d= A* ones(size(A,1),1);
        S= speye(size(A,1))-  sparse(diag(1./sqrt(d))*A*diag(1./sqrt(d)));
        
end
end

