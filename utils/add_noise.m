function Y = add_noise(Y, varargin)

%add_noise Add Gaussian noise to the data according to a specified SNR
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT
%   Y: N x T matrix of noise-free signals
%   SNR_des: desired signal-to-noise ratio (in dB), i.e., 10log10(Py/Pe)
%   seed
%   sigma
%
% OUTPUT
%   Y = noisy data with the specified SNR

    [N,T] = size(Y);
    p = inputParser;
    seed = 0;
    SNR_des = 10;
    
    addParameter(p, 'seed', seed, @isnumeric);
    addParameter(p, 'SNR_des', SNR_des, @isnumeric);
    
    parse(p, varargin{:});
    
    seed = p.Results.seed;
    SNR_des = p.Results.SNR_des;
    
    rng(seed)
   
    E = randn(N,T);
    
    % Compute the power of signal and noise
    Py = (1/(N*T))* trace(Y'*Y);
    Pe = (1/(N*T))* trace(E'*E);
    
    SNR_db = 10*log10(Py/Pe);
    
    disp(['Desired SNR: ' num2str(SNR_des)])
    Pe_new = Py/(10^(SNR_des/10));
    
    E = E * sqrt(Pe_new/Pe);
    Pe = (1/(N*T))* trace(E'*E);
    SNR_db = 10*log10(Py/Pe);
    disp(['Achieved SNR: ' num2str(SNR_db)])

    Y =+ E;

end