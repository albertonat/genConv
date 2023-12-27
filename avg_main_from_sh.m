% Code for the paper "TBD"
%
%
% Currently working.
%
%

%% Remove line below if launching from sh
%clear all; close all % <-- this line
fprintf("Remove clear all and close all if launching from .sh file \n")
pause(1)
%% Configuration
if ispc % Windows machine
    cd('C:\Users\anatali\Documents\Repository\convolution')
elseif isunix % Server
    cd /users/anatali/Desktop/FilterID/gspbox/
    gsp_start
    cd('/users/anatali/Desktop/convolution')
end
thisPath= which('avg_main_from_sh');
ind = strfind(thisPath,'avg_main_from_sh');
pathFolder = thisPath(1:max(1,ind-1));
addpath(genpath(pathFolder));
if ispc, plotFigures=1; else, plotFigures=0; end
if ispc, save_results=1; else, save_results=0; end % save the output matrix
if ~exist('visualizeResults', 'var') && ispc, visualizeResults=1; end
if ~exist('numSimulations', 'var'), numSimulations=2; end % to average results
%% DATA GENERATION

fprintf("Start.\r")
if ~exist('N', 'var'), N= 40; end% # of nodes
if ~exist('T', 'var'), T= 1000; end % # of graph signals (time instants)
if ~exist('L', 'var'), L= 3; end % length filter primal domain
if ~exist('K', 'var'), K=3; end  % length filter dual domain
if ~exist('delta', 'var'), delta=10; end % perturbation jitter
if ~exist('mu', 'var'), mu=1; end % step size descent method
if ~exist('with_mask', 'var'), with_mask=1; else, with_mask=0; end % whether to apply the mask to the expansion coefficient matrix C

NSE_P_simulations= nan(numSimulations,1); % normalized squared error filter
PNE_list_simulations= nan(numSimulations,1); % pascalized error eigenvalues
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---- Primal domain
if ~exist('GSO_type', 'var'), GSO_type="normalized-laplacian"; end
fprintf("GSO_type = %s\n", GSO_type)

G= gsp_sensor(N);
S= getGSO(G.A, GSO_type);
[V,Lambda] = eig(full(S));
lambda= diag(Lambda);
Psi= vandermonde(lambda,L); % Vandermonde primal eigenvalues
% -------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%
for i=1:numSimulations
    currentDateTime = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss');
    fprintf("\n%s Simulation number %d\n", currentDateTime, i)
    seed=i;
    rng(seed);
    X= randn(N,T); % Input data (WGN)
    if with_mask, C= repmat([1:10:10*K]', [1,L]).* randn(K,L); else, C= randn(K,L); end % expansion coefficients, linking primal and dual eigenvalues. Notice
    %that they should depend also from the domain of the polynomials
    

    fprintf("With mask: %d \n", with_mask)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ---- Dual domain
    range = [-1 1]; % domain of the dual eigenvalues
    lambda_f_uniform= linspace(range(1),range(2),N)'; % Uniform locations
    period= (range(2)- range(1))/(N-1);
    opts= struct('type', 'gaussian', 'period', period, 'delta', delta, 'cutoff', 1, 'seed', seed );
    lambda_f = perturbation(lambda_f_uniform, opts); % jittered domain. How it actually is, true sampling locations
    V_f= V';
    S_f= dualGraph(V_f, lambda_f);
    
    Psi_f= vandermonde(lambda_f,K); % Vandermonde dual eigenvalues
    P_hat= Psi*C'; % Dual NV filter taps
    % ------
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % --------- Data and Filters ----------
    % Filters
    P= Psi_f*C; % Primal NV filter taps
    H_I= NV_GF(P,S,"type-I"); % NV-GF primal domain
    H_II= NV_GF(P_hat, S_f, "type-II"); % NV_GF dual domain
    
    % Data
    if ~exist('sigma', 'var'), sigma=0; end% standard deviation noise on the measurements
    if sigma~=0, E= sigma.*randn(N, T); else, E= 0; end
    Y= H_I*X + E;
    Y_GFT= V'*Y;
    Y_f= H_II*(V'*X);
    % --------------------------------------
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Learn taps P
    P_estim= learn_P(X, Y, S, L);
    NSE_P_simulations(i)= (norm(P-P_estim, 'fro')^2)/(norm(P, 'fro')^2);
    
    %% Learn Dual Eigenvalues
    
    params.K=K;
    if ~exist('niter','var'), niter=5000; end % # of SCP iterations
    
    if ~exist('n_starting_points', 'var'), n_starting_points=1; end
    starting_points= initializeStartingPoints(n_starting_points, N, range);
    lambda_f_estim_list= cell(n_starting_points,1); % foreach start_point
    results_list= cell(n_starting_points,1); % foreach start_point
    
    for n=1:n_starting_points % parfor if on server
        plotFlag=0;
        tmp_struct= struct("mu", mu, "niter", niter, "starting_point",...
            starting_points{n}, "plotFlag", plotFlag); % for parfor execution
        [lambda_f_estim, results]= learn_lambda_f(P_estim, tmp_struct);
        lambda_f_estim_list{n}= lambda_f_estim;
        results_list{n}= results;
    end
    
    %% Compute Performance metrics and save the results
    % We compute the NSE and PNSE for all the obtained estimation vectors with
    % the respect the original groundtruth and stack them in vectors. For
    % analysis purposes, we save information about these errors (min, max) in a
    % txt file visible for inspection. We save anyway all the errors for the
    % different starting points.
    %
    z= cell2mat(results_list);
    NE_initial_list= nan(n_starting_points,1);
    NE_list= nan(n_starting_points,1);
    PNE_list= nan(n_starting_points,1);
    
    for n=1:n_starting_points
        lambda_f_estim= lambda_f_estim_list{n};
        NE_initial_list(n)= norm(results_list{n}.starting_point - lambda_f)^2/(norm(lambda_f)^2);
        NE_list(n)= norm(lambda_f_estim- lambda_f)^2/(norm(lambda_f)^2);
        
        % Remove ambiguity for visualization
        % Pascal Procrustes: Find the t_0 t_1 which makes x_hat as close as possible to x
        V2=vandermonde(lambda_f_estim,2);
        t_hat= pinv(V2)*lambda_f;
        
        lambda_f_estim_corrected=  V2*t_hat; % projection onto V2: V2pinv(V2)lambda_f
        PNE_list(n)= (norm(lambda_f_estim_corrected - lambda_f)^2)/((norm(lambda_f)^2));
    end
    
    
    % Select the minimum error achieved among the starting points and store it
    [PNE_list_simulations(i), ~]= min(PNE_list);
end % end simulation i

% Compute the average and median
PNE_struct.mean= mean(PNE_list_simulations);
PNE_struct.median= median(PNE_list_simulations);
PNE_struct.std= std(PNE_list_simulations);

NSE_P_struct.mean= mean(NSE_P_simulations);
NSE_P_struct.median= median(NSE_P_simulations);
NSE_P_struct.std= std(NSE_P_simulations);

output.PNE = PNE_struct;
output.NSE_P = NSE_P_struct;

% Local identifier of the simulation
identifier= sprintf("%s_N%d_K%d_L%d_niter%d_mu%0.2e_delta%0.2e_sigma%0.2e", GSO_type,  N, ...
    K, L,niter, mu, delta, sigma);
identifier= strrep(identifier, '.', 'dot');
if with_mask
    identifierSimulation=fullfile(fullfile(fullfile(pathFolder,'simulations'), sprintf("avg%d_%s", numSimulations, GSO_type)),identifier);
else 
    identifierSimulation=fullfile(fullfile(fullfile(pathFolder,'simulations'),'avg_without_mask'),identifier);
end

% Create a folder to store
if ~exist(identifierSimulation,'dir')
    mkdir(identifierSimulation);
end


file_path= fullfile(identifierSimulation, 'avg.txt' );
fid = fopen(file_path, 'w');
if fid == -1
    error('Cannot open avg.txt file.');
end
fprintf(fid, 'ID: %s \r\n', identifier);
fprintf(fid, 'Total number of simulations: %d \r\n', numSimulations);
fprintf(fid, 'Total number of starting points: %d \r\n', n_starting_points);
fprintf(fid, " NSE filter coefficients P: median = %0.2e,  mean = %0.2e, std= %0.2e \r\n", NSE_P_struct.median,...
   NSE_P_struct.mean, NSE_P_struct.std);
fprintf(fid, " PNE: median = %0.2e,  mean = %0.2e, std= %0.2e \r\n", PNE_struct.median,...
   PNE_struct.mean, PNE_struct.std);
fclose(fid);

% Save the output variable
save(fullfile(identifierSimulation, 'output.mat'), 'output')

fprintf("\r Job Completed: Saved in %s under the name: ----> %s", identifierSimulation,  identifier);



