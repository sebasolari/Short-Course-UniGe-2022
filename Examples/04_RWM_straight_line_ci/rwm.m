function tout = rwm(t0,logPi,logL,nsim)
% Number of parameters
nt   = numel(t0);
% Standard deviation for the Normal proposal function
tstd = .1.*ones(nt,1);
% Initialize output
tout      = zeros(nt,nsim);
tout(:,1) = t0;
logp      = logPi(t0)+logL(t0);
% Run MCMC simulation
for id = 2:nsim
    t_aux    = normrnd(tout(:,id-1),tstd);
    logp_aux = logPi(t_aux)+logL(t_aux);
    if logp_aux>=logp
        tout(:,id) = t_aux;
        logp = logp_aux;
    else
        pacc = exp(logp_aux-logp);
        if pacc>rand
            tout(:,id) = t_aux;
            logp = logp_aux;
        else
            tout(:,id) = tout(:,id-1);
        end
    end
end