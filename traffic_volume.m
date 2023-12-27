%% Traffic Data
%

clear all
close all


cd('C:\Users\anatali\Documents\Repository\convolution')
thisPath= which('main_from_sh');
ind = strfind(thisPath,'main_from_sh');
pathFolder = thisPath(1:max(1,ind-1));
addpath(genpath(pathFolder));


%% Configurations
saveFlag = 1;
GSO_type = 'adjacency';
seed = 3;
method = 'LS';
type_sqrt = 'symmetric';
removeNodes= 1;
name_experiment = 'traffic';
normalize_flag =1 ;
L=3;
K=3;
T = 1259;

%%

load("data\traffic_dataset.mat");

% Create a unique string identifier
identifier = sprintf('method_%s_%s_norm%d_sqrt_%s_GSO_%s_seed%d_L%d_K%d_T%d_removed%d',...
    method, name_experiment, normalize_flag, type_sqrt, GSO_type, seed, L, K, T, removeNodes);

identifierSimulation = fullfile(fullfile(pathFolder, 'simulations/real/traffic/'), identifier)

if saveFlag
    % Create a folder to store
    if ~exist(identifierSimulation,'dir'), mkdir(identifierSimulation); end
end

% Log file metrics
file_path= fullfile(identifierSimulation, 'metrics.txt' );
%%
A = tra_adj_mat;
Y = tra_Y_tr;


%% Preprocessing
% Since the rank of the covariance matrix is deficient, we try to remove some
% nodes (on the straight paths) so that it may eliminate such error. It does not
% actually help.

if removeNodes
    n_remove = [2,3,4, 6,7,8, 9:20, 28,29,7,8, 31, 33, 34];
    %n_remove = [26];
    A(n_remove, :) = [];
    A(:, n_remove) = [];
    Y(n_remove, :) = [];
end
figure
G=graph(A);
a = plot(G);
S = getGSO(A, GSO_type);

coords = [a.XData', a.YData'];

f=figure('color','white');
ax1=axes;

h=plot(ax1,G, 'XData', coords(:,1), 'YData', coords(:,2), 'NodeFontSize', 6,'NodeColor', 'r', 'MarkerSize', 6,...
    'Linewidth',1.5, 'EdgeCData', abs(G.Edges.Weight) ,'EdgeAlpha', 1);
caxis([min(abs(G.Edges.Weight))- 0.0005 max(abs(G.Edges.Weight))])
hold on;
axis off;
ax2=axes;
scatter(ax2, coords(:,1), coords(:,2), 50, 'o','MarkerEdgeColor','k','MarkerFaceColor','r'); hold on;

%scatter(ax2, coords(:,1), coords(:,2), 50, 'o','MarkerEdgeColor','k','MarkerFaceColor','r');
drawnow
s = h.NodeChildren(arrayfun(@(o) isa(o,'matlab.graphics.primitive.world.LineStrip'), h.NodeChildren));
[~,idx] = sortrows(s.ColorData', 'descend');
set(s, 'VertexData',s.VertexData(:,idx),  'ColorData',s.ColorData(:,idx))
linkaxes([ax1,ax2])
ax2.Visible = 'off';
ax1.Visible= 'off';
ax2.XTick = [];
ax2.YTick = [];
colormap(ax2);
%colorbar('southoutside', 'Position', [0.223214285714285,0.076190472471315,0.605357142857143,0.050793650793651])
%%Give each one its own colormap
colormap(ax1,flipud(gray(40)));
set([ax1,ax2],'Position',[.08 .11 .90 .90]);
drawnow
s = h.NodeChildren(arrayfun(@(o) isa(o,'matlab.graphics.primitive.world.LineStrip'), h.NodeChildren));
[~,idx] = sortrows(s.ColorData', 'descend');
set(s, 'VertexData',s.VertexData(:,idx),  'ColorData',s.ColorData(:,idx))

if saveFlag
    saveName= fullfile(identifierSimulation, sprintf("dual_graph"));
    saveas(f,saveName,'png')
    saveas(f,saveName,'fig')
    set(f,'PaperPosition', [0.63 0.89 15.23 11.42], 'PaperSize', [16.5 13.2]);
    print(f,saveName,'-dpdf')
    saveas(f,saveName,'epsc')
    
end




[V,Lambda] = eig(full(S)); % here it gives the orthonormal eigenvectors, so also rescaled lambdas
N= size(A,1);

lag= 2;
%Y= Y(:,2:end) -  Y(:,1:end-1); %differencing
%Y = Y./max(Y,[],2); % normalization
X= Y(:, 1:end-lag);
Y(:,1:lag) = [];
%X= diag(1./randi(5,[N,1]))*Y;

%if strcmp(method, 'LS'), T= 500; end
Y = Y(:, 1:T);
X = X(:, 1:T);
figure
plot(Y')
figure
for i=1:9
    subplot(3,3,i)
    plot(Y(i,:), 'r'), hold on, plot( X(i, :), 'b'); legend('Y', 'X')
    
end
%% Covariance matrix
mY = mean(Y,2);
Cy = (Y-mY)*((Y-mY)')/T;
%Cy = (Y)*(Y)'/T;
if normalize_flag, Y = Y- mY; X = X - mean(X,2); end
figure; imagesc(Cy); colorbar
%% Spectral Covariance
Cy_spec = V'*Cy * V; % Covariance matrix in the frequency domain. If diagonal, Y is stationary on S
imagesc(Cy_spec); colorbar
display(["Stationarity coefficient: " , num2str(stationarity_coefficient(Cy_spec))] );
norm(Cy_spec - diag(diag(Cy_spec)), 'fro')/N

norm(S*Cy - Cy*S,'fro' )
%% Square root
% Make it symmetric. Matlab might have numerical errors
if ~issymmetric(Cy)
    Cy = (Cy + Cy')/2;
end


[Uy,Ey] = eig(Cy);
Ey(abs(Ey)<1e-7)=0;
switch type_sqrt % Which factor F to use for the covariance
    case 'symmetric'
        Cy_root= Uy * sqrtm(Ey) * Uy';
    case 'asymmetric'
        Cy_root= Uy * sqrtm(Ey);
end




a = V'*Uy;
imagesc(a), colorbar % check if the eigenvectors commute
%%
Y_hat = V'*Y;

scatter(1:N, abs(Y_hat(:,6)))
%% Algorithmics

switch method
    case 'LS'
        P_estim= learn_P(X, Y, S, L);
        H_estim = NV_GF(P_estim, S, 'type-I');
        Y_estim = H_estim*X;
        sprintf("Y-Y_estim: %.2e", norm(Y-Y_estim, 'fro')^2/norm(Y,"fro")^2)
    case "LS_ortho"
        
        rng(seed)
        U= eye(N); %RandOrthMat(N, 1e-9); % initialization unitary matrix
        niter=3;
        P_estim= nan(N,L);
        for i=1:niter
            %--- P-step -----
            tic
            %P_estim = learn_P(U*X, gen_Y, S, L);
            P_estim = learn_P(U*X*X', Y*X', S, L);
            toc
            H_I = NV_GF(P_estim, S, 'type-I');
            %sprintf("Iteration %d.1: Objective value = %0.2f", i, norm(gen_Y - H_I*U*X, 'fro')^2/norm(gen_Y, 'fro')^2)
            sprintf("Iteration %d.1: Objective value = %0.2f", i, norm(Y * X' - H_I*U*X*(X'), 'fro')^2/ norm(Y * (X'), 'fro')^2)
            
            %--- U-step -----
            [Wy, ~, Z]=svd(X*Y'*H_I);
            U= Z*Wy';
            
            %sprintf("Iteration %d.2: Objective value = %0.2f", i, norm(gen_Y - H_I*U*X, 'fro')^2/norm(gen_Y, 'fro')^2)
            sprintf("Iteration %d.2: Objective value = %0.2f", i,  norm(Y * X' - H_I*U*X*(X'), 'fro')^2/ norm(Y * (X'), 'fro')^2)
        end
    case 'CM'
        rng(seed)
        U= eye(N);
        %U= RandOrthMat(N, 1e-9); % initialization unitary matrix
        niter=10;
        P_estim= nan(N,L);
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
            
            P_estim= reshape(A\Cy_root(:), [N L]);
            
            
            %---- U-step ------
            H_I= diag(P_estim(:,1));
            S_tmp=speye(N);
            for l=2:L
                S_tmp= S_tmp*S;
                H_I= H_I + diag(P_estim(:,l))*S_tmp;
            end
            
            [Wy, ~, Z]= svd(Cy_root'*H_I);
            U= Z*Wy';
            
            sprintf("Iteration %d: Objective value = %0.2f", i, norm(Cy_root - H_I*U, 'fro')^2/norm(Cy_root , 'fro')^2)
        end
        
end


H_estim = NV_GF(P_estim, S, 'type-I');
Y_estim = [];
nse=nan;
switch  method
    
    % erroe: 3.58e-01 ~ 0.36
    case 'LS'
        Y_estim = H_estim*X;
        figure
        subplot(1,2,1), imagesc(Y_estim); title('Y estimated'); colorbar;
        subplot(1,2,2), imagesc(Y);  title('Y');  colorbar;
        nse = norm(Y - Y_estim, 'fro')^2/norm(Y, 'fro')^2;
        sprintf("Fitting error || Y - Yestim||: %.2e", nse)
        
        % error: 3.48e-03
        %     case 'LS_ortho'
        %         Y_estim = H_estim*U*X;
        %         figure
        %         subplot(1,2,1), imagesc(H_estim*U*X); title('Filter HU'); colorbar;
        %         subplot(1,2,2), imagesc(gen_Y);  title('genY');  colorbar;
        %         nse = norm(gen_Y - H_I*U*X, 'fro')^2/norm(gen_Y, 'fro')^2;
        %         sprintf("Fitting error || Y - HUX||: %.2e", nse)
        %
    case 'CM'
        figure
        subplot(1,2,1), imagesc(H_estim*U); title('Filter HU'); colorbar;
        subplot(1,2,2), imagesc(Cy_root);  title('Square root Covariance Cy');  colorbar;
        nse = norm(Cy_root-H_estim*U, 'fro')^2/norm(Cy_root, 'fro')^2;
        sprintf("Fitting error || Cy root - HU||: %.2e", nse)
end


fid = fopen(file_path, 'w');
if fid == -1
    error('Cannot open metrics.txt file.');
end
fprintf(fid, 'NSE %s: %.4f \r\n', method, nse);
fclose(fid);


if removeNodes
    fid = fopen(file_path, 'a');
    if fid == -1
        error('Cannot open metrics.txt file.');
    end
    fprintf(fid, 'Nodes removed: [%s]\r\n', num2str(n_remove));
    fclose(fid);
end


%% Visualize First Step Inference

f = figure;
imagesc(P_estim);
f.CurrentAxes.XTick = [1:L]; colorbar

if saveFlag
    saveName= fullfile(identifierSimulation, sprintf("P"));
    saveas(f,saveName,'epsc')
    saveas(f,saveName,'png')
    saveas(f,saveName,'fig')
    set(f,'PaperPosition', [0.63 0.89 15.23 11.42], 'PaperSize', [16.5 13.2]);
    print(f,saveName,'-dpdf')
end



%% Learn lambda_f

mu=1;
niter=5000;


n_starting_points=5;
starting_points= initializeStartingPoints(n_starting_points, N, [-1 1]);
lambda_f_estim_list= cell(n_starting_points,1); % foreach start_point
results_list= cell(n_starting_points,1); % foreach start_point

for n=1:n_starting_points % parfor if on server
    tmp_struct= struct("mu", mu, "niter", niter, "starting_point",...
        starting_points{n}, "plotFlag", 1); % for parfor execution
    [lambda_f_estim, results]= learn_lambda_f(P_estim, tmp_struct);
    lambda_f_estim_list{n}= lambda_f_estim;
    results_list{n}= results;
end
%% Dual Graph
V_f = V';
best_idx=0;
best_value= inf;
for i=1: length(results_list)
    if results_list{i}.objvalue(end) < best_value
        best_value = results_list{i}.objvalue(end);
        best_idx=i;
    end
end

lambda_f_estim = lambda_f_estim_list{best_idx};
%lambda_f_estim(abs(lambda_f_estim)<1e-5)=0;

S_f_estim= dualGraph(V_f, lambda_f_estim);

figure
imagesc(S_f_estim); colorbar
S_f_estim * ones(N,1)
figure
scatter(1:N,lambda_f_estim)
%%
%
%% Consistency with the theorem

Psi_f = vandermonde(lambda_f_estim, K);
C = Psi_f \ P_estim;
figure; imagesc(C); title('C'), colorbar
P_hat = vandermonde(diag(Lambda), L)*C';
figure; imagesc(P_hat); title('Phat'); colorbar
H_II = NV_GF(P_hat, S_f_estim, 'type-II');
h = figure; imagesc(H_II); colorbar;  set(h,'Units','normalized','Position',[0 0 1 .5]);
%%

switch method
    case 'LS'
        H_up= V'* H_estim;
        H_down= H_II* V';
        y_up = V'* (H_estim*X);
        y_down = H_II* (V'*X);
        
    case 'LS_ortho'
        y_up = V'* (H_estim* U*X);
        y_down = H_II* (V'*U*X);
    case 'CM'
        H_up= V'* H_estim;
        H_down= H_II* V';
        y_up = V'* (H_estim* U*X);
        y_down = H_II* (V'*U*X);
end

figure;
subplot(2,1,1); imagesc(V*y_up)
subplot(2,1,2); imagesc(V*y_down)

% Approach LS_ortho: 0.0533 ,  LS: 0.0288, CM: 0.027
corollary_error = norm(H_up - H_down, 'fro')^2/ norm(H_up, 'fro')^2




figure
for i=1:9
    subplot(3,3,i)
    plot(y_up(i,:)', 'b'), hold on, plot( y_down(i, :)', 'r'); legend('Y up', 'Y down')
    
end
%% Visualization Dual Graph and Pruning
figure; histogram(S_f_estim, 100, 'Normalization', 'probability')


S_f_estim= dualGraph(V_f, lambda_f_estim);

% Find non-zero elements
nonZeroElements =S_f_estim(S_f_estim ~= 0);
% Determine the threshold value based on the lowest 20% of non-zero elements
thres = prctile(abs(nonZeroElements), 50);
S_f_estim = (S_f_estim + S_f_estim')/2;
S_f_estim(abs(S_f_estim)< thres)=0;
S_f_estim = S_f_estim - diag(diag(S_f_estim));
Gf= graph(S_f_estim);
figure
g = plot(Gf,  "Layout","force", "LineWidth",abs(3*Gf.Edges.Weight/max(Gf.Edges.Weight)));


coords = [g.XData', g.YData'];
%%
f=figure('color','white');

ax1=axes;

h=plot(ax1,Gf, 'XData', coords(:,1), 'YData', coords(:,2), 'NodeFontSize', 6,'NodeColor', 'r', 'MarkerSize', 6,...
    'Linewidth',1.5, 'EdgeCData', abs(Gf.Edges.Weight) ,'EdgeAlpha', 1);
caxis([min(abs(Gf.Edges.Weight))- 0.0005 max(abs(Gf.Edges.Weight))])
hold on;
axis off;
ax2=axes;
scatter(ax2, coords(1:floor(end/2),1), coords(1:floor(end/2),2), 50, 'o','MarkerEdgeColor','k','MarkerFaceColor',[0.153, 0.714, 1]); hold on;
scatter(ax2, coords(floor(end/2)+1:end,1), coords(floor(end/2)+1:end,2), 50, 'o','MarkerEdgeColor','k','MarkerFaceColor','r');

%scatter(ax2, coords(:,1), coords(:,2), 50, 'o','MarkerEdgeColor','k','MarkerFaceColor','r');
drawnow
s = h.NodeChildren(arrayfun(@(o) isa(o,'matlab.graphics.primitive.world.LineStrip'), h.NodeChildren));
[~,idx] = sortrows(s.ColorData', 'descend');
set(s, 'VertexData',s.VertexData(:,idx),  'ColorData',s.ColorData(:,idx))
linkaxes([ax1,ax2])
ax2.Visible = 'off';
ax1.Visible= 'off';
ax2.XTick = [];
ax2.YTick = [];
colormap(ax2);
%colorbar('southoutside', 'Position', [0.223214285714285,0.076190472471315,0.605357142857143,0.050793650793651])
%%Give each one its own colormap
colormap(ax1,flipud(gray(40)));
set([ax1,ax2],'Position',[.08 .11 .90 .90]);
drawnow
s = h.NodeChildren(arrayfun(@(o) isa(o,'matlab.graphics.primitive.world.LineStrip'), h.NodeChildren));
[~,idx] = sortrows(s.ColorData', 'descend');
set(s, 'VertexData',s.VertexData(:,idx),  'ColorData',s.ColorData(:,idx))

if saveFlag
    saveName= fullfile(identifierSimulation, sprintf("dual_graph"));
    saveas(f,saveName,'png')
    saveas(f,saveName,'fig')
    set(f,'PaperPosition', [0.63 0.89 15.23 11.42], 'PaperSize', [16.5 13.2]);
    print(f,saveName,'-dpdf')
    saveas(f,saveName,'epsc')
    
end


fid = fopen(file_path, 'a');
if fid == -1
    error('Cannot open metrics.txt file.');
end
fprintf(fid, 'corollary_error: %.3f \r\n', corollary_error);

fclose(fid);



%% Clustering


S_f_estim= dualGraph(V_f, lambda_f_estim);

% Remove self loops
S_f_estim = S_f_estim - diag(diag(S_f_estim));
S_f_estim = abs(S_f_estim);
S_f_estim = (S_f_estim+S_f_estim')/2;
D_f = diag(S_f_estim * ones(N,1));
L_f = D_f - S_f_estim;
[uf,sf] = eig(L_f);
embedding = uf(:,1:3);
figure; scatter3(embedding(:,1), embedding(:,2), embedding(:,3))
%%
% We do the same but with the builtin Matlab function
[idx, v] = spectralcluster(S_f_estim, 3,'Distance','precomputed','LaplacianNormalization','symmetric');
figure
scatter(1:N, idx)

