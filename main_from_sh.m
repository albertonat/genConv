% Code for the paper "A General Convolution Theorem and its Relations
%                       to Non-Stationarity and Graph Frequency Domain"
%
% Currently working.
%
%

%% Remove line below if launching from sh
clear all; close all % <-- this line
fprintf("Remove clear all and close all if launching from .sh file \n")
pause(1)
%% Configuration
if ispc % Windows machine
    cd('C:\Users\anatali\Documents\Repository\convolution')
elseif isunix % Server
    cd /users/anatali/Desktop/FilterID/cvx-a64/cvx/
    cvx_setup
    cd /users/anatali/Desktop/FilterID/gspbox/
    gsp_start
    cd('/users/anatali/Desktop/convolution')
end
thisPath= which('main_from_sh');
ind = strfind(thisPath,'main_from_sh');
pathFolder = thisPath(1:max(1,ind-1));
addpath(genpath(pathFolder));
if ispc, plotFigures=1; else, plotFigures=0; end
if ispc, save_results=0; else, save_results=0; end % save the output matrix
if ~exist('visualizeResults', 'var') && ispc, visualizeResults=1; end

%% DATA GENERATION

fprintf("Start.\r")
if ~exist('N', 'var'), N= 20; end% # of nodes
if ~exist('T', 'var'), T= 3000; end % # of graph signals (time instants)
if ~exist('L', 'var'), L=9 ; end % length filter primal domain
if ~exist('K', 'var'), K=9; end  % length filter dual domain
if ~exist('delta', 'var'), delta=10; end % perturbation jitter


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---- Primal domain
if ~exist('GSO_type', 'var'), GSO_type='normalized-adjacency'; end
G= gsp_sensor(N);
S= getGSO(G.A, GSO_type);
[V,Lambda] = eig(full(S));
lambda= diag(Lambda);
Psi= vandermonde(lambda,L); % Vandermonde primal eigenvalues
% -------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%
% seed =1 for L=K=3 and both sigma
% seed 6 for L=K=9
if ~exist('seed', 'var'), seed=6; end
rng(seed)
X= randn(N,T); % Input data (WGN)
C= repmat([1:10:10*K]', [1,L]).* randn(K,L);% expansion coefficients, linking primal and dual eigenvalues. Notice
%that they should depend also from the domain of the polynomials

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---- Dual domain
range = [-1 1]; % domain of the dual eigenvalues
lambda_f_uniform= linspace(range(1),range(2),N)'; % Uniform locations
period= (range(2)- range(1))/(N-1);
opts= struct('type', 'gaussian', 'period', period, 'delta', delta, 'cutoff', 1, 'seed', seed, 'plot', 1);
lambda_f = perturbation(lambda_f_uniform, opts); % jittered domain. How it actually is, true sampling locations
%lambda_f= randn(N,1);
V_f= V';
S_f= dualGraph(V_f, lambda_f);

Psi_f= vandermonde(lambda_f,K); % Vandermonde dual eigenvalues
P_hat= Psi*C'; % Dual NV filter taps
% ------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --------- Data and Filters ----------
% Filters
P= Psi_f*C; % Primal NV filter taps
H_I= NV_GF(P,S,"type-I"); % NV-GF primal domain
H_II= NV_GF(P_hat, S_f, "type-II"); % NV_GF dual domain

% Data
if ~exist('sigma', 'var'), sigma=0; end% standard deviation noise on the measurements
if sigma~=0, E= sigma.*randn(N, T); else, E= 0; end
Y= H_I*X + E;


% CONTINUE FROM HERE 
Y = add_noise(Y, 'seed', 3, 'SNR_des',40);
% Just to check the SNR with different levels of sigma
if 0
    sigma = 50;
    E= sigma.*randn(N, T);
    P_Y = 1/N * diag(Y'*Y);
    P_E = 1/N * diag(E'*E);
    
    SNR_db = 10*log10(P_Y./P_E);
    
    % If I would compute using the trace I would basically do also a temporal
    % average
    P_Y = (1/(N*T)) * trace(Y'*Y);
    P_E = (1/(N*T)) * trace(E'*E);
    SNR_db = 10*log10(P_Y./P_E);
    
    SNR_des = 1; % desired SNR
    
    P_E_new = P_Y/10^(SNR_des/10)
    
    E_new = E* sqrt(P_E_new / P_E);
    P_E_new = (1/(N*T)) * trace(E_new'*E_new);
    SNR_db = 10*log10(P_Y./P_E_new);
end

Y_GFT= V'*Y;
Y_f= H_II*(V'*X);
% --------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% In this case I am directly generating the function values without
% actually sampling any polynomial. But we can show which are the actual
% polynomials
plotFlag=0;
%if plotFlag, plotPolynomials(range, C, lambda_f_uniform,vandermonde(lambda_f_uniform,K)*C, lambda_f, P, 1 ); end

if plotFigures
    figure
    subplot(1,2,1); imagesc(S); title("Primal")
    subplot(1,2,2); imagesc(S_f); title("Dual")
    
    Ry= (1/T)*(Y*Y'); % empirical covariance matrix
    figure, subplot(1,2,1), imagesc(Ry), xlabel('Empirical R_y')
    subplot(1,2,2), imagesc(H_I*H_I'), xlabel('Theoretical R_y')
    % This confirms that the filter implementation is correct.
end



%% Learn taps P
fprintf(" o o o o o o Learning the filter tap matrix P o o o o o o \r")
if ~exist('useInput', 'var'), useInput= true; end
P_estim= learn_P(X, Y, S, L, useInput);
NSE_P=norm(P-P_estim, 'fro')^2/norm(P, 'fro')^2;
fprintf("----> NSE P= %1.3e\r -----------\r\r", NSE_P)

%pause
%% Learn Dual Eigenvalues
fprintf(" //////////// Learning the dual eigenvalues lambda_f //////////////// \r")

params.K=K;
niter=5000; % # of SCP iterations.
mu=1; % step size descent method

if ~exist('n_starting_points', 'var'), n_starting_points=10; end
starting_points= initializeStartingPoints(n_starting_points, N, range);
lambda_f_estim_list= cell(n_starting_points,1); % foreach start_point
results_list= cell(n_starting_points,1); % foreach start_point

for n=1:n_starting_points % parfor if on server
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
    PNE_list(n)= norm(lambda_f_estim_corrected - lambda_f)^2/((norm(lambda_f)^2));
end


% Compute statistics to save in a text file for quick inspection
NE_mean= mean(NE_list); NE_std= std(NE_list);
PNE_mean= mean(PNE_list); PNE_std= std(PNE_list); PNE_median= median(PNE_list);
NE_initial_mean= mean(NE_initial_list); NE_initial_std= std(NE_initial_list);

% Local identifier of the simulation
identifier= sprintf("%s_N%d_K%d_L%d_niter%d_mu%0.2e_delta%d_sigma%0.2e", GSO_type,  N, ...
    K, L,niter, mu, delta, sigma);
identifier= strrep(identifier, '.', 'dot');
identifierSimulation=fullfile(pathFolder,'simulations',identifier);

% Create a folder to store the log, results and figures
if ~exist(identifierSimulation,'dir')
    mkdir(identifierSimulation);
end


file_path= fullfile(identifierSimulation, 'avg.txt' );
fid = fopen(file_path, 'w');
if fid == -1
    error('Cannot open avg.txt file.');
end
fprintf(fid, 'Seed: %d ; Total number of starting points: %d \r\n', seed, n_starting_points);
fprintf(fid, " NSE filter coefficients P: = %0.2e \r\n", NSE_P);
fprintf(fid, " Mean --->  NE = %0.2e NEi = %0.2e PNE = %0.2e \r\n", NE_mean, NE_initial_mean, PNE_mean);
fprintf(fid, " Std --->  NE = %0.2e NEi = %0.2e PNE = %0.2e \r\n", NE_std, NE_initial_std, PNE_std);
fprintf(fid, " Median --->   PNE = %0.2e \r\n  Min:%0.2e \r\n", PNE_median, min(PNE_list));
fclose(fid);


output.errors.PNE_list= PNE_list;
output.errors.NE_list= NE_list;
output.errors.NE_initial_list= NE_initial_list;
output.lambda_f_estim_list=lambda_f_estim_list;
output.lambda_f= lambda_f;
output.V_f=V_f;
output.starting_points= starting_points;


% Save the output variable
if save_results, save(fullfile(identifierSimulation, 'output.mat'), 'output');end


%% Visualization
if ispc
    saveFigure=1;
    folderImages= identifierSimulation;
    [v, index_lambda_f_best]= min(output.errors.PNE_list);
    fprintf("Best PNE: %0.2e , index: %i", v, index_lambda_f_best);
    lambda_f_best= output.lambda_f_estim_list{index_lambda_f_best}; % notice that the saved lambda are not corrected
    S_f_estim= dualGraph(V_f, lambda_f_best);
    
    V2=vandermonde(lambda_f_best,2);
    t_hat= pinv(V2)*output.lambda_f;
    lambda_f_best_corrected=  V2*t_hat; % projection onto V2
    S_f_estim_corrected= dualGraph(V_f, lambda_f_best_corrected);
    
    if plotFigures
        f=figure('color','white');
        subplot(1,3,1); imagesc(S_f); title("Dual"); colorbar
        subplot(1,3,2); imagesc(S_f_estim); title("Dual (NC)"); colorbar
        subplot(1,3,3); imagesc(S_f_estim_corrected); title("Dual Inferred (Corrected)"); colorbar
    end
    
    
    f=figure('color','white');
    imagesc(S_f); title("Dual"); colorbar
    if saveFigure
        saveName= fullfile(folderImages, sprintf("dual_K%dL%d_delta%d_sigma%d", K,L, delta,sigma));
        saveas(f,saveName,'epsc')
        saveas(f,saveName,'fig')
        set(f,'PaperPosition', [0.63 0.89 15.23 11.42], 'PaperSize', [16.5 13.2]);
        print(f,saveName,'-dpdf') 
    end
    
    f=figure('color','white');
    imagesc(S_f_estim); title("Dual (NC)"); colorbar
    if saveFigure
        saveName= fullfile(folderImages, sprintf("dual_inferred_NC_K%dL%d_delta%d_sigma%d", K,L, delta, sigma));
        saveas(f,saveName,'epsc')
        saveas(f,saveName,'fig')
        set(f,'PaperPosition', [0.63 0.89 15.23 11.42], 'PaperSize', [16.5 13.2]);
        print(f,saveName,'-dpdf')
    end
    close
    
    
    if saveFigure
        f=figure('color','white');
        imagesc(S_f_estim_corrected); title("Dual Inferred (Corrected)"); colorbar
        saveName= fullfile(folderImages, sprintf("dual_inferred_C_K%dL%d_delta%d_sigma%d", K,L, delta, sigma));
        saveas(f,saveName,'epsc')
        saveas(f,saveName,'fig')
        set(f,'PaperPosition', [0.63 0.89 15.23 11.42], 'PaperSize', [16.5 13.2]);
        print(f,saveName,'-dpdf')
    end
    
    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %                      Scatterplot eigenvalues                           %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    starting_point_best= output.starting_points{index_lambda_f_best};
    if plotFigures
        f=figure('color','white');
        scatter(1:N, starting_point_best, 'd', 'filled'); hold on;
        scatter(1:N, output.lambda_f, 'og', 'filled');
        %scatter(1:N, lambda_f_best, 30,'*c');
        scatter(1:N, lambda_f_best_corrected, 30,'*r');
        legend("initial", "true", "estimated (C)", 'location','northwest')
        %legend("initial", "true", "estimated (NC)", "estimated (C)", 'location','northwest')
        xlabel('Index'), ylabel('Value');
        saveName= fullfile(folderImages, sprintf("locations_K%dL%d_delta%d_sigma%d", K,L, delta, sigma));
        saveas(f,saveName,'epsc')
        saveas(f,saveName,'fig')
        set(f,'PaperPosition', [0.63 0.89 15.23 11.42], 'PaperSize', [16.5 13.2]);
        print(f,saveName,'-dpdf')
    end
    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %                         RECONSTRUCTED POLYNOMIALS                      % 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Psi_f_hat_corrected= vandermonde(lambda_f_best_corrected,K);
            C_hat_corrected= pinv(Psi_f_hat_corrected)*P;
            P_estim_corrected= Psi_f_hat_corrected * C_hat_corrected; % it is equal to the P only if 0 error on cost
            

            msize = 30; % marker size
            linwid = 1.5; % line width
            colors={[0 0.4470 0.7410],[0.8500 0.3250 0.0980], [0.9290 0.6940 0.1250],...
                [0.4940 0.1840 0.5560], [0.4660 0.6740 0.1880], [0.3010 0.7450 0.9330],...
                [0.6350 0.0780 0.1840], [1 0 1], [0 1 1] };
            polynomials_hat= cell(L,1);
            polynomials= cell(L,1);
            syms t
            figure
            legendTitle = cell(1,L);
            for i = 1:L
                legendTitle{1,i} = sprintf('f_{%d}', i);
            end
            hold on
            for l=1:L
                rrr= colors{l};
                polynomials{l} = C(:,l)'*t.^(0:K-1).';
                %polynomials_hat{l} = W_hat(:,l)'*t.^(0:K-1).'; % this is with the ambiguity
                polynomials_hat_corrected{l} = C_hat_corrected(:,l)'*t.^(0:K-1).'; % this is without the ambiguity
                fplot(polynomials{l}, [min(lambda_f) max(lambda_f)], 'Color',rrr,'LineWidth',linwid)
                fplot(polynomials_hat_corrected{l}, [min(lambda_f_best_corrected) max(lambda_f_best_corrected)], 'Color','k','LineWidth',linwid,'LineStyle', "--",'HandleVisibility','off')                
                %scatter(x_hat,Y_hat(:,l), msize + 5, '*', 'MarkerEdgeColor',rrr, 'MarkerFaceColor', rrr,'HandleVisibility','off') % this is with the ambiguity
                scatter(lambda_f_best_corrected,P_estim_corrected(:,l), msize + 5, '*', 'MarkerEdgeColor',rrr, 'MarkerFaceColor', rrr,'HandleVisibility','off')
                scatter(lambda_f,P(:,l), msize, 'o','MarkerEdgeColor',rrr,  'HandleVisibility','off');
            end
            legend(legendTitle);  grid on
            xlabel('x'), ylabel('y')
    
end
fprintf("\r Job Completed.")

