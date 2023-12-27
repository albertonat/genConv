function Sf = dualGraph(V,lambda)
%DUALGRAPH Creates a (dual) graph
%   V      : (dual) eigenvectors
%   lambda : (dual) eigenvalues
    Sf= sparse(V*diag(lambda)* V');
end

