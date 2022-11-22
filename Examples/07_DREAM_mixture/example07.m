% Generates nx samples
t  = [.3;0;.5;1;.25];
nx = 200;
x  = normrnd(t(4),t(5),nx,1);
n1 = round(t(1)*nx);
x(1:n1) = normrnd(t(2),t(3),n1,1);

% Defines functions
logPi = @(tt) logPi_aux(tt);
logL  = @(tt) max(-Inf,...
    sum(log( tt(1).*normpdf(x,tt(2),tt(3)) + ...
    (1-tt(1)).*normpdf(x,tt(4),tt(5)) )));

% Parameters for running DREAM(ZS)
par.d   = 5;           % Number of model parameters
par.nc  = 3;           % Number of chains
par.upb = [ 1  1 1  1 1];  % Upper limit for initial sample
par.lob = [ 0 -1 0 -1 0];  % Lower limit for initial sample

% Runs DREAM(ZS)
out     = dreamzs(logL,logPi,par);

% Plot chains
figure
subplot(2,1,1); plot(out.z);
subplot(2,1,2); plot(out.xr,out.r);

% Burnin data (with R2)
nbi = out.xr(find(sum(out.r<1.2,2)<par.d,1,'last')+1);
% Mean value of the parameters
t_mean = mean(out.z(nbi:end,:),1);
[~,idm] = max(out.p);
t_maxl  = out.z(idm,:);

% Plot MCMC evolution and histograms
figure
for id = 1:par.d
    subplot(par.d,2,2*(id-1)+1); plot(out.z(:,id))
    subplot(par.d,2,2*id);
    histogram(out.z(nbi:end,id),'Normalization','pdf'); hold on;
    plot(t(id),0,'o',t_maxl(id),0,'s',t_mean(id),0,'^'); hold off;
    legend('histogram rwm','true','max logL','mean rwm');
end
figure
for id = 1:par.d
    subplot(par.d,1,id);
    plot(reshape(out.z(:,id),numel(out.z(:,id))/par.nc,par.nc),'.');
end



function lp = logPi_aux(tt)
if tt(1)>1 || tt(1)<0 || tt(3)<0 || tt(5)<0
    lp = -Inf;
else
    lp = 0;
end
end