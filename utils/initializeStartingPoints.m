function starting_points= initializeStartingPoints(n, N, range)
%starting_points Generate n starting_points of dimension Nx1 in the range
%range(1) and range(2). The values are in ascending order and generated
%from a uniform distribution.
%
%
% INPUT
%   - n: (scalar) number of starting points of the algorithm
%   - N: (scalar) number of entries of each point
%   - range: (1 x 2 vector) min and maximum admissible values eigenvalues
%
% OUTPUT
%   - starting_points: (cell)
if ~exist('seed', 'var'), seed=1; end
rng(seed)
pd = makedist('Uniform', 'Lower', range(1), 'Upper', range(2));
starting_points = cell(n,1);

starting_points{1}= flip(linspace(-1,1,N)');
% starting_points{1}= ones(N,1); <- just for debugging out of the space
%starting_points{1}= linspace(range(1),range(2),N)'; % uniform starting points
starting_points{2}= randn(N,1);

for i=3:n
    rng(i)
    %starting_points{i}= sort(pd.random(N,1), 'ascend');
    starting_points{i}= pd.random(N,1);
end

%fprintf(" ----- Generated %d starting points for the algorithm. o o o o o\r", n);
end