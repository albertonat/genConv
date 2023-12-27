
if ~exist('pathFolder', 'var')
    pathFolder = "Z:\Desktop\convolution";
    addpath(genpath(pathFolder));
end

% Save Path
folderImages= 'C:\Users\anatali\Documents\Repository\convolution\figures';


% Where the numerical simulation are stored
%dirSimulations= fullfile(pathFolder, 'simulations\avg50');
%dirSimulations= fullfile(pathFolder, 'simulations\avg_without_mask');
dirSimulations= fullfile(pathFolder, 'simulations\avg50_normalized-adjacency');
listResults= dir(dirSimulations);

if listResults(1).name=='.'
    listResults([1,2])=[];
end

files = {listResults.name}.';

sigmaToText = @(x) strrep(sprintf("sigma%0.2e",x), '.', 'dot');
deltaToText = @(x) strrep(sprintf("delta%0.2e",x), '.', 'dot');

delta_list=[1 10 100 1000];
sigma_list= [0 0.5 5];



linwid= 1.5;
colors={[0 0.4470 0.7410],[0.8500 0.3250 0.0980], [0.9290 0.6940 0.1250],...
       [0.4940 0.1840 0.5560], [0.4660 0.6740 0.1880], [0.3010 0.7450 0.9330], [0.6350 0.0780 0.1840], 'k' };
legendTitle = cell(1,7);
for i = 3:9
    legendTitle{1,i-2} = sprintf('L=K=%d', i);
end

%% Plot PNE as a function of delta for a fixed sigma
f1= figure(); hold on; grid on;
sigma=0; % fixed sigma <-- change this for a different sigma
sigmaText= sigmaToText(sigma);
for k=3:9
    l=k;
    rows= cellfun(@(x) contains(x,strcat("K", int2str(k))) & contains(x,strcat("L", int2str(l))) & contains(x,sigmaText) , files);
    filteredFiles = listResults(rows);
    nFiles= length(filteredFiles);
    tmp= nan(nFiles,1);
    for j=1:nFiles
        %results = load(fullfile(filteredFiles(j).name,'output.mat')).('output');
        results = load(fullfile(dirSimulations,fullfile(filteredFiles(j).name,'output.mat'))).('output');
        tmp(j) = results.PNE.median;
    end
    plot(20*log10(tmp), 'LineWidth',linwid, 'Color', colors{k-2}, 'Marker', '*')
end
if sigma==0
legend(legendTitle, 'Position', [0.75,0.30,0.15,0.24]);
elseif sigma==0.5 || sigma==5 || sigma==50
    legend(legendTitle, 'location', 'southwest');
end
f1.CurrentAxes.XTickLabel([2,4,6])= cell(3,1);
f1.CurrentAxes.XTickLabel{1}= '1';
f1.CurrentAxes.XTickLabel{3}= '10';
f1.CurrentAxes.XTickLabel{5}= '100';
f1.CurrentAxes.XTickLabel{7}= '1000';
xlabel("\delta", 'Fontsize', 15); ylabel('PNE (dB)');
saveName= fullfile(folderImages, sprintf("PNE_over_delta_with_sigma%s", sigmaText));
saveas(f1,saveName,'epsc')
saveas(f1,saveName,'fig')
set(f1,'PaperPosition', [0.63 0.89 15.23 11.42], 'PaperSize', [16.5 13.2]);
print(f1,saveName,'-dpdf')



%% Plot PNE as a function of delta for a fixed sigma


P =[1 4 2 3];

% Plot with xlabel equal to sigma
f2= figure(); hold on;
grid on
%for d=1:length(delta_list)
%delta=delta_list(d)
delta=1000; % fixed delta <-- change this for a different delta
deltaText = deltaToText(delta);
for k=3:9
    l=k;
    rows= cellfun(@(x) contains(x,strcat("K", int2str(k))) & contains(x,strcat("L", int2str(l))) & contains(x,deltaText)  , files);
    filteredFiles = listResults(rows);
    nFiles= length(filteredFiles);
    tmp= nan(nFiles,1);
    for j=1:nFiles
        %results= load(fullfile(filteredFiles(P(j)).name,'output.mat')).('output');
        results= load(fullfile(dirSimulations,fullfile(filteredFiles(P(j)).name,'output.mat'))).('output');
        tmp(j)=results.PNE.median;
    end
    plot([1,2,3,4], 20*log10(tmp), 'LineWidth',linwid, 'Color', colors{k-2}, 'Marker', '*')
end
legend(legendTitle, 'location', 'southeast');
f2.CurrentAxes.XTickLabel([2,4,6])= cell(3,1);
f2.CurrentAxes.XTickLabel{1}= '0';
f2.CurrentAxes.XTickLabel{3}= '0.5';
f2.CurrentAxes.XTickLabel{5}= '5';
f2.CurrentAxes.XTickLabel{7}= '50';
%f2.CurrentAxes.XLabel.Position= [2.5,-609,-1];
xlabel("\sigma", 'Fontsize', 15); ylabel('PNE (dB)');
saveName= fullfile(folderImages, sprintf("PNE_over_sigma_with_delta%d", delta));
saveas(f2,saveName,'epsc')
saveas(f2,saveName,'fig')
set(f2,'PaperPosition', [0.63 0.89 15.23 11.42], 'PaperSize', [16.5 13.2]);
print(f2,saveName,'-dpdf')


 
%end
