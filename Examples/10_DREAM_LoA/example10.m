% Generates 100 samples
% f = @(tt,x) tt(1).*x+tt(2);
% x = linspace(0,10,100)';
% t = [.5;5];
% y = f(t,x) + normrnd(0,1,numel(x),1);

% Defines functions
logPi = @(tt) 0;
LoA   = 3;
logL  = @(tt) sum(abs(y-f(tt,x))<LoA);

% Parameters for running DREAM(ZS)
par.d   = 2;           % Number of model parameters
par.nc  = 3;           % Number of chains
par.upb = [ 2 10];  % Upper limit for initial sample
par.lob = [ 0  0];  % Lower limit for initial sample

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

% Plot results
figure
plot(x,f(t,x),'k-'); hold on;
plot(x,y,'ro');
plot(x,f(t_mean,x),'r--'); hold off;

% Least-square estimation for comparison
[t_ls,~,r] = regress(y,[x ones(size(x))]);
t_ls       = [t_ls;std(r)];

% Plot MCMC evolution and histograms
figure
for id = 1:par.d
    subplot(par.d,2,2*(id-1)+1); plot(out.z(:,id))
    subplot(par.d,2,2*id);
    histogram(out.z(nbi:end,id),'Normalization','pdf'); hold on;
    plot(t(id),0,'o',t_ls(id),0,'s',t_mean(id),0,'^'); hold off;
    legend('histogram rwm','true','least squares','mean rwm');
end
figure
for id = 1:par.d
    subplot(par.d,1,id);
    plot(reshape(out.z(:,id),numel(out.z(:,id))/par.nc,par.nc),'.');
end

% Plot IC
nboot   = 1e4;
samp_t  = out.z(randsample(nbi:end,nboot,'true'),:);
y_out   = samp_t(:,1)'.*x+samp_t(:,2)';
y_ic    = quantile(y_out',[.05 .95]);
figure
plot(x,f(t,x),'k-'); hold on;
plot(x,y,'ro');
plot(x,y_ic','r:'); hold off;
