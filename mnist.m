clear all
close all


cd('C:\Users\anatali\Documents\Repository\convolution')
thisPath= which('main_from_sh');
ind = strfind(thisPath,'main_from_sh');
pathFolder = thisPath(1:max(1,ind-1));
addpath(genpath(pathFolder));

%% Configurations
name_experiment = 'mnist';
normalize_flag = 1;
type_sqrt = 'symmetric';
GSO_type='normalized-laplacian';
plotFlag = 0;
saveFlag = 1;
digit=5; % select a digit from 0 to 9
seed = 5;

L=4; % length primal filter
K=4; % length dual filter
T=100000; % number of graph signals before optimization (for the covariance matrix of the new signals)
method = 'LS'; % Algorithm to learn the filter taps P. Choices: 'LS', 'LS_ortho', 'CM'
if strcmp(method,'LS'), T_alg=1000; else, T_alg=T; end

% Create a unique string identifier with 'digit' and 'method' parameters first
identifier = sprintf('digit%d_method_%s_%s_norm%d_sqrt_%s_GSO_%s_seed%d_L%d_K%d_T%d',...
    digit, method, name_experiment, normalize_flag, type_sqrt, GSO_type, seed, L, K, T);

identifierSimulation = fullfile(fullfile(pathFolder, 'simulations/real/mnist'), identifier)
% Create a folder to store
if saveFlag
    if ~exist(identifierSimulation,'dir'), mkdir(identifierSimulation); end
end

% Log file metrics
file_path= fullfile(identifierSimulation, 'metrics.txt' );

%% Load Dataset
resized=1;
if resized
    load('train-images-18x18.mat');
else
    filenameImagesTrain = 'train-images-idx3-ubyte.gz';
    Y_original = processImagesMNIST(filenameImagesTrain); % H x W x 1 x numImages
    Y= squeeze(Y_original);
end
filenameLabelsTrain = 'train-labels-idx1-ubyte.gz';
labels = double(string(processLabelsMNIST(filenameLabelsTrain)));



%% Select Digit

idx_digit= find(digit==labels);
fprintf('\nThe digit %d occurs %d times\n', digit, length(idx_digit));
Y_digit= Y(:,:, idx_digit);
[H, W, numImages_digit]=size(Y_digit);

% Show images of the selected digit
figure
for i=1:9
    subplot(3,3,i); imshow(Y_digit(:,:,i), [])
end


%% Preprocessing

Y_flat= reshape(Y_digit, [H*W, numImages_digit]);
mY = mean(Y_digit, 3);
%mY=0;
if plotFlag, figure, imagesc(Y_flat); title('Original'), colorbar; end

if normalize_flag
    %mY(mY<1e-3)=0
    if plotFlag, figure; heatmap(mY); title("Mean"); end
    Y_flat = ((Y_flat - mY(:)));
    if plotFlag, figure, imagesc(Y_flat); title('Normalized'), colorbar; end
    
    Cy_digit= (Y_flat * Y_flat')/numImages_digit; % mean is already removed
    
else
    Cy_digit= ((Y_flat - mY(:)) * (Y_flat - mY(:))')/numImages_digit;
end

if plotFlag
    figure
    title('Normalized Digits'),
    for i=1:9
        subplot(3,3,i); imshow(reshape(Y_flat(:,i), [H W]), [])
    end
end

%% Covariance matrix
% Make it symmetric. Matlab might have numerical errors
if ~issymmetric(Cy_digit)
    Cy_digit = (Cy_digit + Cy_digit')/2;
end
if plotFlag, figure; imagesc(Cy_digit); colorbar, end

[Uy, Dy]= eig(Cy_digit);
Dy(abs(Dy)<1e-7)=0;


% We can try to use different matrix decomposition of Cy as long as it is
% written as Cy= FF' for some F


switch type_sqrt % Which factor F to use for the covariance
    case 'symmetric'
        Cy_digit_root= Uy * sqrtm(Dy) * Uy';
    case 'asymmetric'
        Cy_digit_root= Uy * sqrtm(Dy);
end
if plotFlag, figure; imagesc(Cy_digit_root); title('Square root Cy'), colorbar, end

%% Data Generation

rng(seed)
N= H*W; % number of nodes
X = randn(N, T);
gen_Y = Cy_digit_root * X;

figure;
for i=1:9
    subplot(3,3,i); imshow(reshape(gen_Y(:,i), [H W]), [])
end
sgtitle("Created through covariance factor")

gen_Cy = ((gen_Y - mean(gen_Y, 2)) *(gen_Y - mean(gen_Y, 2))')/ size(gen_Y,2); % in principle no need to remove the mean
if plotFlag
    figure;
    subplot(1,2,1),  imagesc(gen_Cy); title('Cy generated'),colorbar
    subplot(1,2,2), imagesc(Cy_digit); title('Cy'), colorbar
end

%% Graph

% Create coordinates
n=18;
m=n;
coords = [reshape(repmat((0:(m-1))/m,n,1),m*n,1), flip(repmat((0:(n-1))'/n,m,1))];
% Grid Graph
G = gsp_2dgrid(H); % notice that it creates a graph H x H
G.coords = coords;
param.show_edges = 1;
if plotFlag, figure, gsp_plot_signal(G, gen_Y(:,6)), end


% Graph Shift Operator
S= getGSO(G.W, GSO_type); % typical of a grid-graph
[V,Lambda] = eig(full(S));
if plotFlag, figure; imagesc(S), end


%% Stationarity

Cy_digit_spec = V'*gen_Cy * V;
if plotFlag, figure, imagesc(Cy_digit_spec); title('Spectral Cy'), colorbar, end
display(["Stationarity coefficient: " , num2str(stationarity_coefficient(Cy_digit_spec))] );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                  Algorithmics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X= X(:,1:T_alg);
gen_Y = gen_Y(:,1:T_alg);

% FIT P

switch method
    % Least Squares
    case 'LS'
        tic
        P_estim= learn_P(X, gen_Y, S, L);
        toc
        
        % Least Squares with Orthogonal matrix
    case 'LS_ortho'
        rng(seed)
        U= eye(N); %RandOrthMat(N, 1e-9); % initialization unitary matrix
        niter=3;
        P_estim= nan(N,L);
        for i=1:niter
            %--- P-step -----
            tic
            %P_estim = learn_P(U*X, gen_Y, S, L);
            P_estim = learn_P(U*X*X', gen_Y*X', S, L);
            toc
            H_I = NV_GF(P_estim, S, 'type-I');
            %sprintf("Iteration %d.1: Objective value = %0.2f", i, norm(gen_Y - H_I*U*X, 'fro')^2/norm(gen_Y, 'fro')^2)
            sprintf("Iteration %d.1: Objective value = %0.2f", i, norm(gen_Y * X' - H_I*U*X*(X'), 'fro')^2/ norm(gen_Y * (X'), 'fro')^2)
            
            %--- U-step -----
            [Wy, ~, Z]=svd(X*gen_Y'*H_I);
            U= Z*Wy';
            
            %sprintf("Iteration %d.2: Objective value = %0.2f", i, norm(gen_Y - H_I*U*X, 'fro')^2/norm(gen_Y, 'fro')^2)
            sprintf("Iteration %d.2: Objective value = %0.2f", i, norm(gen_Y * X' - H_I*U*X*(X'), 'fro')^2/ norm(gen_Y * (X'), 'fro')^2)
        end % end niter A-M
        
        % Covariance Matching
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
            
            P_estim= reshape(A\Cy_digit_root(:), [N L]);
            
            
            %---- U-step ------
            H_I= diag(P_estim(:,1));
            S_tmp=speye(N);
            for l=2:L
                S_tmp= S_tmp*S;
                H_I= H_I + diag(P_estim(:,l))*S_tmp;
            end
            
            [Wy, ~, Z]= svd(Cy_digit_root'*H_I);
            U= Z*Wy';
            
            sprintf("Iteration %d: Objective value = %0.2f", i, norm(Cy_digit_root - H_I*U, 'fro')^2/norm(Cy_digit_root , 'fro')^2)
        end % end niter A-M
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
        subplot(1,2,2), imagesc(gen_Y);  title('YXt');  colorbar;
        nse = norm(gen_Y - Y_estim, 'fro')^2/norm(gen_Y, 'fro')^2;
        sprintf("Fitting error || genY - Yestim||: %.2e", nse)
        
        % error: 3.48e-03
    case 'LS_ortho'
        Y_estim = H_estim*U*X;
        figure
        subplot(1,2,1), imagesc(H_estim*U*X); title('Filter HU'); colorbar;
        subplot(1,2,2), imagesc(gen_Y);  title('genY');  colorbar;
        nse = norm(gen_Y - H_I*U*X, 'fro')^2/norm(gen_Y, 'fro')^2;
        sprintf("Fitting error || Y - HUX||: %.2e", nse)
        
    case 'CM'
        Y_estim = H_estim*U*X;
        figure
        subplot(1,2,1), imagesc(H_estim*U); title('Filter HU'); colorbar;
        subplot(1,2,2), imagesc(Cy_digit_root);  title('Square root Covariance Cy');  colorbar;
        nse = norm(Cy_digit_root-H_estim*U, 'fro')^2/norm(Cy_digit_root, 'fro')^2;
        sprintf("Fitting error || Cy root - HU||: %.2e", nse)
end


fid = fopen(file_path, 'w');
if fid == -1
    error('Cannot open metrics.txt file.');
end
fprintf(fid, 'NSE %s: %.3f \r\n', method, nse);
fclose(fid);


%% Visualize First Step Inference
figure
for i=1:9
    subplot(3,3,i); imshow(reshape(Y_estim(:,i), [H W]), []) % add mean?
end
sgtitle("Y estimated after learning P")

f= figure; %sgtitle("Filter taps estimated");
cbax = axes('Position', [0.82,0.099,0.046,0.79]);
caxis([min(min(P_estim)) max(max(P_estim))])

for i=1:L
    subplot(2,2,i)
    imshow(reshape(P_estim(:, i), [H, W]), []);
    caxis([min(min(P_estim)) max(max(P_estim))]);
    
    %subplot(2,2,i); imagesc(reshape(P_estim(:, i), [H, W])); colormap(gray), axis image
    %pos = get(gca, 'Position');
    %pos = [0.2000 0.5563 0.3947 0.3981];
    %pos = [0.5100 0.5538 0.3947 0.4012]; % second tap
    %pos = [0.2000 0.2000 0.3347 0.3412]
    %set(gca, 'Position', pos);
end
ha = get(f, 'children');
set(ha(1), 'position', [.5 .1 .3 .4])
set(ha(2),'position',[.1 .1 .5 .4])
set(ha(3),'position',[.5 .5 .3 .4])
set(ha(4),'position',[.1 .5 .5 .4])

if saveFlag
    saveName= fullfile(identifierSimulation, sprintf("P"));
    saveas(f,saveName,'epsc')
    saveas(f,saveName,'png')
    saveas(f,saveName,'fig')
    set(f,'PaperPosition', [0.63 0.89 15.23 11.42], 'PaperSize', [16.5 13.2]);
    print(f,saveName,'-dpdf')
end




%% Learn lambda_f

mu=100;
niter=2000;


n_starting_points=3;
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

figure;
imagesc(S_f_estim)

figure;
scatter(1:N,lambda_f_estim)

%% Theorem Check
Psi_f = vandermonde(lambda_f_estim, K);
C = Psi_f \ P_estim;
if plotFlag, figure; imagesc(C); colorbar; end
P_hat = vandermonde(diag(Lambda), L)*C';
if plotFlag, figure; imagesc(P_hat); colorbar, end

if plotFlag
    for i=1:K
        subplot(2,3,i); imagesc(reshape(P_hat(:,i), [H W])); colorbar
    end
end

H_II = NV_GF(P_hat, S_f_estim, 'type-II');
h = figure; imagesc(H_II); colorbar;  set(h,'Units','normalized','Position',[0 0 1 .5]);


% Watch out to y_up here since we need to consider for the orthogonal
% matrix for approach 2 and 3.
% Approach



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
subplot(2,1,1); imagesc(y_up)
subplot(2,1,2); imagesc(y_down)


f = figure;
for i=1:9
    subplot(3,3,i); imagesc(reshape(V*y_up(:,i), [H W])); colormap(gray), axis off
end
sgtitle("Y through primal")

if saveFlag
    saveName= fullfile(identifierSimulation, sprintf("Y_up"));
    saveas(f,saveName,'epsc')
    saveas(f,saveName,'png')
    saveas(f,saveName,'fig')
    set(f,'PaperPosition', [0.63 0.89 15.23 11.42], 'PaperSize', [16.5 13.2]);
    print(f,saveName,'-dpdf')
end

% Showing some y_down converted to primal domain
f= figure;
for i=1:9
    subplot(3,3,i); imagesc(reshape(V*y_down(:,i), [H W])); colormap(gray), axis off
end
sgtitle("Y through dual")

if saveFlag
    saveName= fullfile(identifierSimulation, sprintf("Y_down"));
    saveas(f,saveName,'epsc')
    saveas(f,saveName,'png')
    saveas(f,saveName,'fig')
    set(f,'PaperPosition', [0.63 0.89 15.23 11.42], 'PaperSize', [16.5 13.2]);
    print(f,saveName,'-dpdf')
end

% Approach LS_ortho: 0.0533 ,  LS: 0.0288, CM: 0.027
corollary_error = norm(H_up - H_down, 'fro')^2/ norm(H_up, 'fro')^2


%% Visualization Dual Graph and Pruning
figure; histogram(S_f_estim, 1000, 'Normalization', 'probability')

S_f_estim= dualGraph(V_f, lambda_f_estim);

% Find non-zero elements
nonZeroElements =S_f_estim(S_f_estim ~= 0);
% Determine the threshold value based on the lowest 20% of non-zero elements
thres = prctile(abs(nonZeroElements), 98);
S_f_estim = (S_f_estim + S_f_estim')/2;
S_f_estim(abs(S_f_estim)< thres)=0;
S_f_estim = S_f_estim - diag(diag(S_f_estim));

Gf= graph(S_f_estim);
figure
g = plot(Gf, "Layout","force");




%%%%%%%%%
f=figure('color','white');

ax1=axes;
coords = [g.XData' g.YData'];

h=plot(ax1,Gf, 'XData', coords(:,1), 'YData', coords(:,2), 'NodeFontSize', 6,'NodeColor', 'r', 'MarkerSize', 7,...
    'Linewidth',1.5, 'EdgeCData',Gf.Edges.Weight ,'EdgeAlpha', 1);
caxis([min(Gf.Edges.Weight) max(Gf.Edges.Weight)])
hold on;
axis off;
ax2=axes;
scatter(ax2, coords(1:floor(end/2),1), coords(1:floor(end/2),2), 50, 'o','MarkerEdgeColor','k','MarkerFaceColor',[0.153, 0.714, 1]); hold on;
scatter(ax2, coords(floor(end/2)+1:end,1), coords(floor(end/2)+1:end,2), 50, 'o','MarkerEdgeColor','k','MarkerFaceColor','r');

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
% for k = 1:N
%         text(coords(k,1), coords(k,2),['  ' num2str(k)],'Color','k', ...
%             'FontSize',7,'FontWeight','b')
% end
for i=1:N
    % Offset to avoid overlap
    offset = 0.05;
    
    % Add text annotations with dynamic positioning to avoid overlap
    textPosition = [coords(i,1), coords(i,2)] + offset;
    
    % Check for overlap and adjust position if needed
    while any(pdist2(textPosition, [coords(:,1), coords(:,2)]) < 2 * offset)
        textPosition = textPosition + offset;
    end
    
    text(textPosition(1), textPosition(2), num2str(i), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize',7,'FontWeight','b');
end

if saveFlag
    saveName= fullfile(identifierSimulation, sprintf("dual_graph"));
    saveas(f,saveName,'epsc')
    saveas(f,saveName,'png')
    saveas(f,saveName,'fig')
    set(f,'PaperPosition', [0.63 0.89 15.23 11.42], 'PaperSize', [16.5 13.2]);
    print(f,saveName,'-dpdf')
    
end


fid = fopen(file_path, 'a');
if fid == -1
    error('Cannot open metrics.txt file.');
end
fprintf(fid, 'corollary_error: %.3f \r\n', corollary_error);

fclose(fid);


%% New Data
XX = randn(N, 10);
YY = H_estim*XX;

f = figure;
for i=1:9
    subplot(3,3,i); imagesc(reshape(YY(:,i) - mY(:), [H W])); colormap(gray), axis off
end
sgtitle("New Y")

if saveFlag
    saveName= fullfile(identifierSimulation, sprintf("new_Y"));
    saveas(f,saveName,'epsc')
    saveas(f,saveName,'png')
    saveas(f,saveName,'fig')
    set(f,'PaperPosition', [0.63 0.89 15.23 11.42], 'PaperSize', [16.5 13.2]);
    print(f,saveName,'-dpdf')
end

