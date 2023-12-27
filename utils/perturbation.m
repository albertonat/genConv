function [y, varargout] = perturbation(x, opts)
%perturbation Apply a perturbation to the data
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT
%   x: data
%   type: textual specification of the perturbation type
%       - "jitter":

%   varargin{1}: in the case of "jitter", it represents the smallest
%   interval among two consecutive samples in the vector x. In the case of
%   uniform sampling, it corresponds to the sampling period. Half of this
%   quantity is the maximum admissible jitter that can be added to a
%   sample without creating overlapping with the adjacents. It is used as
%   the radius of a ball centered at a point.
%   varargin{2}:  in the case of "jitter", it represents the scaling factor
%   applied to varargin{1}, so the extension of the centered ball.
%
%
% OUTPUT
%   y = perturbed data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


N= size(x,1);
switch opts.type
    case 'jitter'
        %period= varargin{1};
        %scaleFactor= varargin{2};
        rball= opts.delta*(opts.period/2);
        pd= makedist("Normal");
        t= truncate(pd, -rball, rball);
        jitter= random(t,N,1);
        y= x+jitter;
        
    case 'gaussian'
        % Check for parameters
        if (~isfield(opts,'period'))
            error("The variable period in opts has not been initialized.")
        end
        if (~isfield(opts,'delta'))
            opts.delta=1;
        end
        if (~isfield(opts,'cutoff'))
            opts.cutoff=0; % no cut-off
        end
        
        
        sigma= opts.delta*(opts.period/2); % standard deviation Gaussian
        pd = makedist('normal','mu',0,'sigma',sigma);
        if opts.cutoff~=0
            assert(sigma~=0, "Error. The standard deviation of the perturbation should be greater than 0. Check that delta is greater than 0.")
            pd=truncate(pd, -opts.cutoff*sigma, opts.cutoff*sigma);
        end
        
        % Just to plot the distribution
        if (isfield(opts,'plot'))
            d=[-3:0.001:3]; 
            y=pdf(pd,d);
            plot(d,y);
            hold on;
            scatter(x, zeros(N,1)); % uniform locations
        end
        
        if (isfield(opts, 'seed')), rng(opts.seed); end
        j= random(pd,N,1);
        y= x+j;
        
    case 'none'
        y=x;
end



end

