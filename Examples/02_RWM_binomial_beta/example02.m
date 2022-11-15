% Ejemplo binomial/beta con Metropolis
a0       = 1;   b0       = 1;
n_test   = 16;  n_exitos = 13;
p_std    = .1;  nsim     = 1e4;
p        = zeros(nsim,1);
p(1)     = betarnd(a0,b0);
logL0    = n_exitos*log(p(1))+(n_test-n_exitos)*log(1-p(1))+log(betapdf(p(1),a0,b0));
for I = 2:nsim
    p_aux = -1;
    while p_aux>1 || p_aux<0
        p_aux     = normrnd(p(I-1),p_std);
    end
    logL_aux  = n_exitos*log(p_aux)+(n_test-n_exitos)*log(1-p_aux)+log(betapdf(p_aux,a0,b0));
    if logL_aux>=logL0
        p(I)  = p_aux;
        logL0 = logL_aux;
    else
        r = exp(logL_aux-logL0);
        u = unifrnd(0,1);
        if u<=r
            p(I)  = p_aux;
            logL0 = logL_aux;
        else
            p(I)  = p(I-1);
        end
    end
end
figure
x0 = 0:.01:1;
plot(x0,betapdf(x0,a0,b0),x0,betapdf(x0,a0+n_exitos,n_test-n_exitos+b0));
hold on; histogram(p(nsim/10:end),'Normalization','pdf')
