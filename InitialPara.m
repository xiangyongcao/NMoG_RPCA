function [prior, model] = InitialPara(param,muOn,B)
% Description: initialize prior parameters and model parameters of NMoG_RPCA model.
% Input:
% muOn        ----    The switch for updating mu. 
%                     muOn = 1 : update mu; 
%                     muOn = 0 : set mu as 0 without updating.

prior.alpha0 = 1e-3;     prior.beta0 = 1e3;      prior.mu0 = 1e-4;
prior.c0 = 1e-3;         prior.eta0 = 1e-3;
prior.lambda0 = 1e-3;    prior.a0 = 1e-6;        prior.b0 = 1e-6;

model.alpha = 1e-3*ones(param.mog_k,B);     model.beta = 1e-3*ones(param.mog_k,B);
model.mu = 0*ones(param.mog_k,B);           model.c = 1e-3*ones(param.mog_k,B);
model.d = 1e-3*ones(param.mog_k,B);         model.eta = 1e-3;
model.lambda = 1e-3;                        model.R = [];
model.muOn = muOn;