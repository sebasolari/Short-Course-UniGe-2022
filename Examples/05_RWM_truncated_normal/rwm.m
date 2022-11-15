function [tout,lout] = rwm(t0,logPi,logL,nsim)
% Number of parameters
nt   = numel(t0);
% Standard deviation for the Normal proposal function
tstd = .1.*ones(nt,1);
% Initialize output
tout      = zeros(nt,nsim);
lout      = zeros(nsim,1);
tout(:,1) = t0;
lout(1)   = logPi(t0)+logL(t0);
% Run MCMC simulation
for id = 2:nsim
    t_aux    = normrnd(tout(:,id-1),tstd);
    logp_aux = logPi(t_aux)+logL(t_aux);
    if logp_aux>=lout(id-1)
        tout(:,id) = t_aux;
        lout(id)   = logp_aux;
    else
        pacc = exp(logp_aux-lout(id-1));
        if pacc>rand
            tout(:,id) = t_aux;
            lout(id)   = logp_aux;
        else
            tout(:,id) = tout(:,id-1);
            lout(id)   = lout(id-1);
        end
    end
end