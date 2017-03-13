function model = VariatInf(model,prior,E_YminusUV, E_YminusUV_2)
% Variational inference of NMoG_RPCA. 

model = nmog_vmax(model, prior, E_YminusUV,E_YminusUV_2);
model = nmog_vexp(model, E_YminusUV,E_YminusUV_2);


function model = nmog_vmax(model,prior,Ex,Ex2)
alpha0 = prior.alpha0;
beta0  = prior.beta0;
mu0    = prior.mu0;
muOn   = model.muOn;
c0     = prior.c0;
R      = permute(model.R,[1,3,2]);
[m,n]  = size(Ex);
k      = size(R,3);
for i=1:k
    nxbar(i,:) = diag(Ex'*R(:,:,i));
    temp(i,:)  = diag(Ex2'*R(:,:,i));
end
nk           = reshape(sum(R,1),n,k);
model.alpha  = alpha0 + nk';
model.beta   = beta0 + nk';
model.c      = c0 + nk'/2;
model.mu     = muOn*(beta0*mu0+nxbar) ./ model.beta;
model.d      =  model.eta/model.lambda + 0.5*( temp + beta0.*(mu0.^2) - 1 ./ model.beta.*(nxbar + beta0.*mu0).^2 );
model.eta    = prior.eta0 + k*n*prior.c0;
model.lambda = prior.lambda0 + sum(sum(model.c ./ model.d));

function  model = nmog_vexp(model, Ex, Ex2)
alpha = model.alpha;
beta  = model.beta;
mu    = model.mu;
c     = model.c;
d     = model.d;
[m,n] = size(Ex);
k     = size(mu,1);
tau   = c./d;
EQ    = zeros(m,n, k);
Elogtau = psi(0, c) - log(d);
Elogpi  = psi(0, alpha) - psi(0, sum(alpha(:)));
for i=1:k
    temp = bsxfun(@times,tau(i,:),Ex2) - 2*bsxfun(@times,tau(i,:).*mu(i,:),Ex);
    EQ(:,:,i) = bsxfun(@plus,1./beta(i,:) + tau(i,:).*mu(i,:).^2 ,temp);
    logRho(:,:,i) = (bsxfun(@minus,EQ(:,:,i),2*Elogpi(i,:) + Elogtau(i,:) - log(2*pi)))/(-2);
end
logR = bsxfun(@minus,logRho,logsumexp(logRho,3));
R = exp(logR);
model.logR = logR;
model.R = permute(R,[1,3,2]);

