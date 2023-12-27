function H= NV_GF(taps,S, type)
%NV-GF Node-Variant Graph Filter
 
% INPUT
%   -taps: matrix of coefficients N x length
%   -S: GSO N x N
%   -type: type-I or type-II

% OUTPUT
%   -H: Type-I or type-II NV-GF

[N,length]= size(taps);
H= diag(taps(:,1));
S_acc= speye(N);
for l=1:length-1
    S_acc= S_acc*S;
    switch type
        case "type-I"
            H= H +  diag(taps(:,l+1))* S_acc;
        case "type-II"
            H= H +  S_acc*diag(taps(:,l+1));   
    end
end

end





