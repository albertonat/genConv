function [lambda_f_estim, results]= learn_lambda_f(P, params)
%fitPolynomials Fits the model to the data
% It implements the SCP method of the paper to find the variable x
% whose Vandermonde matrix is maximum aligned with the data subspace.

%
%  INPUT
%  - P: N X L matrix containing data to fit. Each column is a sampled
%       polynomial
%  - params: structure containing at least the following fields
%       - starting_point: initial starting point of the algorithm
%       - niter: number of iterations
%       - mu: step size
%
% OUTPUT
%
% - lambda_f_estim: N x 1 vector of estimated locations
% - results: structure containing additional results


N=size(P,1);
niter = params.niter; % SCP iterations
lambda_f_estim= params.starting_point;
mu= params.mu;


% Signal Subspace
[U,Sigma,~]= svd(P);
sgvalues= find(diag(Sigma)>1e-14);
Us= U(:,sgvalues); % signal subspace


rem=0;
if rem
%%%%%  ADDED PART 26/01/23
one= (1/sqrt(N))* ones(N,1);
one_pro= Us*pinv(Us)*one; % vector of ones is in the subspace
Urem= (eye(N) - one*one')*Us;
x= Urem*pinv(Urem)*one;
(eye(N) - Urem*pinv(Urem))*one; % now 1 is on the subspace 
f= f_subspace(Urem); % function object
%%%
else
f= f_subspace(Us); % function object
end

startIndx = 1; endIndx = 0;
objvalue= nan(niter,1); gradNorm= nan(niter,1); objvalue(1)= f.eval(lambda_f_estim);

g_iter=nan(N,niter); % gradient variable over iterations
x_iter=nan(N,niter); % x variable over iterations


%% Sequential Convex Programming routine
for i=2:niter
    
    gradf= f.grad(lambda_f_estim);
    gradNorm(i-1)= norm(gradf);
    
    x_feasible= lambda_f_estim- mu* gradf;
    
    % Grid search
    dsc_dir=x_feasible - lambda_f_estim; % descent direction
    alpha= 0:0.0025:1;
    cost_original= zeros(length(alpha),1); % function evaluation
    cost_approximation=zeros(length(alpha),1); % approximate function evaluation
    
    for ia=1:length(alpha)
        a= alpha(ia);
        x_cvx= lambda_f_estim + a*dsc_dir; % convex combination
        cost_original(ia)= f.eval(x_cvx);
        cost_approximation(ia)= objvalue(i-1) + transpose(gradf)*(x_cvx- lambda_f_estim);
    end

    [~, minInd]= min(cost_original);
    x_optimal= lambda_f_estim + alpha(minInd)*dsc_dir;
    %sprintf(" ---> New Best Function Value: %3.4f ", f.eval(x_optimal))
    lambda_f_estim= x_optimal;
    
    
    objvalue(i)=  f.eval(lambda_f_estim);

    if i>3 
        if x_iter(:, i-2) == x_iter(:,i-3)
        mu=0.9*mu;
        %fprintf("Adjusting step size mu: %0.2e \r\n", mu)
        end 
    end
    % Plotting real-time
    if mod(i,1000) == 0 && params.plotFlag
        
        figure(100)
        endIndx = i;
        vecIndx = startIndx:endIndx;
        plot(vecIndx, 10*log10(objvalue(vecIndx)),'-b','LineWidth',2);
        hold on
        ylabel('Function Value'); xlabel('Iteration Number');
        legend('Function value')
        
        
        figure(200)
        plot(vecIndx, 10*log10(gradNorm(vecIndx)),'-r','LineWidth',2);
        hold on
        ylabel('Gradient Norm'); xlabel('Iteration Number');
        legend('grad')
        pause(1/48)
        
        startIndx = endIndx + 1;
    end
    
    g_iter(:,i-1)= gradf;
    x_iter(:,i-1)=lambda_f_estim;  
end
fprintf("Final step-size mu: %.2e \n", mu)

results.objvalue=objvalue;
results.g_iter= g_iter;
results.x_iter=x_iter;
results.x_hat=lambda_f_estim;
results.starting_point= params.starting_point;

end