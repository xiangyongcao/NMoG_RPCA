function [model,lr_model] = NMoG_RPCA(Y,Rank,param,model,prior)
% non_i.i.d. MoG Robust Principal Components Analysis
% Inputs:
%    Y          ----  input data matrix
%    Rank       ----  objective rank of the low-rank component
%    param      ----  input parameters
%       param.maxiter      : number of iterations allowed (default: 100)
%       param.tol          : stop criterion (default: 1e-6)
%       param.mog_k        : number of Gaussians in the noise component (default: 3)
%       param.initial_rank : initial rank of the low-rank component (default: full rank)
%       param.rankDeRate   : number of rank reduced in each iteration (default: 1)
%       param.lr_init      : method for initializing the low-rank component
%                             'SVD'  : using SVD (default)
%                             'rand' : random initialization
%    prior      ----  hyperparameters of the non i.i.d. MoG RPCA model
%    model      ----  initial posteriori parameters of the non i.i.d. MoG RPCA
%
% Outputs:
%    model      ---- estimated posteriori parameters of the non i.i.d. MoG RPCA
%    lr_model   ---- estimated model parameters of the low-rank component

% more detail can be found in
% [1] Chen, Y., Cao, X., Zhao, Q., Meng, D., & Xu, Z. (2017). Denoising Hyperspectral Image with Non-iid Noise Structure.
% arXiv preprint arXiv:1702.00098.
%
% @article{chen2017denoising,
%   title={Denoising Hyperspectral Image with Non-iid Noise Structure},
%   author={Chen, Yang and Cao, Xiangyong and Zhao, Qian and Meng, Deyu and Xu, Zongben},
%   journal={arXiv preprint arXiv:1702.00098},
%   year={2017}
% }
%
% Written by Yang Chen (if you have any questions/comments/suggestions, please contact me: yangchen9103@gmail.com)

if (~isfield(param,'maxiter'))
    maxiter = 100;
else
    maxiter = param.maxiter;
end

if (~isfield(param,'tol'))
    tol = 1e-6;
else
    tol = param.tol;
end

if (~isfield(param,'mog_k'))
    mog_k = 3;
else
    mog_k = param.mog_k;
end

if (~isfield(param,'lr_init'))
    lr_init = 'SVD';
else
    lr_init = param.lr_init;
end

if (~isfield(param,'initial_rank'))
    initial_rank = min(size(Y));
else
    initial_rank = param.initial_rank;
end

if (~isfield(param,'rankDeRate'))
    rankDeRate = 1;
else
    rankDeRate = param.rankDeRate;
end

if (~isfield(param,'display'))
    display = 1;
else
    display = param.display;
end

clear param;
[N,B] = size(Y);
k = mog_k;

% Initial low-rank component
Y2sum = sum(Y(:).^2);
scale = sqrt( Y2sum / (N*B));
if strcmp(lr_init, 'SVD')   % SVD initialization
    [u, s, v] = svd(Y, 'econ');
    r = initial_rank;
    U = u(:,1:r)*(s(1:r,1:r)).^(0.6);
    V = (s(1:r,1:r)).^(0.4)*v(:,1:r)';
    V = V';
    Sigma_U = repmat( scale*eye(r,r), [1 1 N] );
    Sigma_V = repmat( scale*eye(r,r), [1 1 B] );
    gammas = 1/scale^2*ones(r,1);
elseif strcmp(lr_init, 'rand')  % Random initialization
    r = initial_rank;
    U = randn(N,r) * sqrt(scale);
    V = randn(B,r) * sqrt(scale);
    Sigma_U = repmat( scale*eye(r,r), [1 1 N] );
    Sigma_V = repmat( scale*eye(r,r), [1 1 B] );
    gammas = scale*ones(r,1);
end
lr_model.U = U;
lr_model.V = V;
lr_model.Sigma_U = Sigma_U;
lr_model.Sigma_V = Sigma_V;
lr_model.gammas = gammas;


%%%%%%%%% Initial model parameters %%%%%%%%%%%%%%%
E = Y-U*V';
for i=1:B
    model.R(:,:,i) = R_initialization(E(:,i)', k);
    alpha0 = prior.alpha0;
    beta0 = prior.beta0;
    mu0 = prior.mu0;
    c0 = prior.c0;
    nxbar = reshape(E(:,i), 1,N)*model.R(:,:,i);
    nxbar = nxbar';
    nk = sum(model.R(:,:,i),1)';
    model.alpha(:,i) = alpha0+nk;
    model.beta(:,i) = beta0+nk;
    model.c(:,i) = c0+nk/2;
    model.mu(:,i) = model.muOn*(beta0.*mu0+nxbar)./model.beta(:,i);
    temp = reshape(E(:,i).^2, 1, N)*model.R(:,:,i);
    model.d(:,i) =  model.eta/model.lambda +0.5*( temp' + beta0.*(mu0.^2) -1./model.beta(:,i).*(nxbar+beta0.*mu0).^2 );
end

% main loop
i = 1;  converged = 0;
while ~converged && i < maxiter + 1
    E_old = E;
    
    % Low rank component update
    [lr_model, E_YminusUV, E_YminusUV_2] = lr_update(Y,Rank,lr_model,model,prior,rankDeRate);
       
    % NMoG component update
    model = VariatInf(model,prior,E_YminusUV, E_YminusUV_2);
    
    % Convergence check
    E = Y-lr_model.U*lr_model.V';
    converged = norm(E-E_old,'fro')/norm(E_old,'fro') < tol;
    
    if display
        r = size(lr_model.U,2);
        disp(['The rank in ',num2str(i) ' iteration is ',num2str(r),';']);
        if converged == 1
            disp(' converged.');
        else
            disp(' not converged.');
        end
    end
    i = i+1;
end


%%
function R = R_initialization(X, k)
n = size(X, 2);
idx = randsample(n,k);
m = X(:,idx);
[~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
[u,~,label] = unique(label);
while k ~= length(u)
    idx = randsample(n,k);
    m = X(:,idx);
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
    [u,~,label] = unique(label);
end
R = full(sparse(1:n,label,1,n,k,n));

%%
function [lr_model, E_YminusUV, E_YminusUV_2] = lr_update(Y, Rank,lr_model, model, lr_prior,rankDeRate)
[N, B]  = size(Y);
Sigma_U = lr_model.Sigma_U;
Sigma_V = lr_model.Sigma_V;
gammas  = lr_model.gammas;
a0 = lr_prior.a0;
b0 = lr_prior.b0;
U  = lr_model.U;
V  = lr_model.V;
R  = model.R;
c  = model.c;
d  = model.d;
mu = model.mu;
k  = size(mu,1);
r  = size(U,2);
tau = c./d;
Gam = diag(gammas);
Rtau    = reshape( sum(repmat(reshape(tau,1,k,B),N,1).*R,2), N,B);
Rtaumu  = reshape( sum(repmat(reshape(tau.*mu,1,k,B),N,1).*R,2), N,B);
RtauYmu = Rtau.*Y - Rtaumu;

% Update U
re_Sigma_V = reshape(Sigma_V, r*r, B);
diagsU = zeros(r,1);
temp_U = zeros(r,r,N);
for i=1:N
    Sigma_U(:,:,i) = ( reshape( re_Sigma_V*Rtau(i,:)', r, r ) + bsxfun(@times, V', Rtau(i,:))*V + Gam )^(-1);
    U(i,:) = (RtauYmu(i,:)*V) * Sigma_U(:,:,i);
    diagsU = diagsU + diag( Sigma_U(:,:,i) );
    temp_U(:,:,i) = Sigma_U(:,:,i)+U(i,:)'*U(i,:);  % <U(i,:)'*U(i,:)>
end

% Update V
re_Sigma_U = reshape(Sigma_U, r*r, N);
diagsV = zeros(r,1);
temp_V = zeros(r,r,B);
for j=1:B
    Sigma_V(:,:,j) = ( reshape( re_Sigma_U*Rtau(:,j), r, r ) + bsxfun(@times, U', Rtau(:,j)')*U + Gam )^(-1);
    V(j,:) = (RtauYmu(:,j)'*U) * Sigma_V(:,:,j);
    diagsV = diagsV + diag( Sigma_V(:,:,j) );
    temp_V(:,:,j) = Sigma_V(:,:,j)+V(j,:)'*V(j,:);
end

% Update gammas
gammas = ( 2*a0 + N + B )./( 2*b0 + sum(U.*U)'+ diagsU + sum(V.*V)' + diagsV);

% Update Rank
Len = length(gammas);
if Len>Rank
    [v,l] = sort(gammas);
    indices = l(1:max(Len-rankDeRate,Rank));
    U = U(:,indices);
    V = V(:,indices);
    lr_model.gammas = gammas(indices);
    lr_model.Sigma_U = Sigma_U(indices,indices,:);
    lr_model.Sigma_V = Sigma_V(indices,indices,:);
    temp_U = temp_U(indices,indices,:);
    temp_V = temp_V(indices,indices,:);
end
r  = size(U,2);
E_YminusUV = Y - U*V';
E_YminusUV_2 = Y.^2 - 2.*Y.*(U*V') + reshape(reshape(temp_U,r*r,N)'*reshape(temp_V,r*r,B),N,B);
lr_model.U = U;
lr_model.V = V;



