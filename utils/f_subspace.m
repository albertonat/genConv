classdef f_subspace < handle % reference semantics, not value semantics
    
    properties
        U % subspace matrix
        N % # of locations/variables
        K % order of the polynomials
        D_k % supDiag(1,2,..., K-1), K x K matrix
        P_U % Projection matrix complement of U
    end
    
    methods
        
        function obj= f_subspace(U)
            obj.N= size(U,1);
            obj.K=size(U,2);
            obj.D_k= diag(1: obj.K-1,1);
            obj.U=U;
            obj.P_U= speye(obj.N) - U*pinv(U);
        end
        
        function M= compute_M(obj, x)
            V= vandermonde(x, obj.K); % the assumption here is that the subspace of P (its rank) reveals the dimension of the Vandermonde
            M=obj.P_U*V;
        end
        
        
        function v = eval(obj, x)
            M= obj.compute_M(x);
            v= 0.5*norm(M,'fro')^2;
        end
        
        function gradf = grad(obj,x, varargin)
            V= vandermonde(x,obj.K);
            M= obj.compute_M(x);
            VDk= V*obj.D_k;
            gradf= diag(obj.P_U'*M*VDk');
            if nargin==3
                fprintf("  ---- Gradient norm: %3.4f ", norm(gradf))
            end
        end
        
        
        
    end
end
