function plotPolynomials(range, C, lambda_f_uniform, P_uniform, lambda_f, P, seed, varargin)
%PLOTPOLYNOMIALS Summary of this function goes here
%   Detailed explanation goes here

rng(seed)
if numel(varargin)==2
    msize= varargin{1} ;
    linwid = varargin{2};
elseif numel(varargin)==1
    msize = varargin{1}; % marker size
    linwid = 1.5; % line width
elseif numel(varargin)==0
    msize = 30; % marker size
    linwid = 1.5; % line width
end
N= size(P,1);
L= size(P,2);
K= size(C,1);
polynomials= cell(L,1);
syms t
legendTitle = cell(1,L);
for i = 1:L
    legendTitle{1,i} = sprintf('f_{%d}', i);
end
colors={[0 0.4470 0.7410],[0.8500 0.3250 0.0980], [0.9290 0.6940 0.1250],...
    [0.4940 0.1840 0.5560], [0.4660 0.6740 0.1880], [0.3010 0.7450 0.9330], [0.6350 0.0780 0.1840]...
    [1 0 0],[0 1 0],[0 0 1],	[0 1 1],[1 1 0],	[1 0 1], [0 0 0] };
figure
hold on
for l=1:L
    rrr= colors{l};
    polynomials{l} = C(:,l)'*t.^(0:K-1).';
    fplot(polynomials{l}, [range(1)-.5 range(2)+.5], 'Color',rrr,'LineWidth',linwid);
    scatter(lambda_f,P(:,l), msize, 'o','MarkerEdgeColor',rrr, 'MarkerFaceColor', rrr, 'HandleVisibility','off');
    %scatter(lambda_f_uniform,P_uniform(:,l),msize, 'o','MarkerEdgeColor',rrr, 'HandleVisibility','off' );
    scatter(3*ones(N,1), P(:,l),20,'MarkerEdgeColor',colors{l}, 'MarkerFaceColor', colors{l}, 'HandleVisibility','off')
    xlim([range(1)-.5 range(2)+3])
    ylim([min(min(P))-5  max(max(P))+5])
end
tex = text(1.7607,-0.53482,'\rightarrow','FontSize',18,'Color','r');
xlabel('x'), ylabel('y')
legend(legendTitle, 'orientation', 'horizontal');
line([lambda_f_uniform, lambda_f_uniform], ylim, 'Color', [.7 .7 .7], 'LineWidth', 0.3,'LineStyle','--','HandleVisibility','off');

end

