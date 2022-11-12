% Generates 100 samples
f = @(tt,x) tt(1).*x+tt(2);
x = linspace(0,10,100)';
t = [.5;5;1];
y = f(t,x) + normrnd(0,t(3),numel(x),1);

% Defines functions
logPi = @(tt) 0;
logL  = @(tt) -numel(y)*log(tt(3)) - .5*sum( ( (f(tt,x)-y)./tt(3) ).^2 );

% Run RWM
nsim = 1e5;
tout = rwm([0;0;1],logPi,logL,nsim);

% Burnin data (estimated)
nbi = round(0.2*nsim);
% Mean value of the parameters
t_mean = mean(tout(:,nbi:end),2);

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
subplot(3,2,1); plot(tout(1,:))
subplot(3,2,2);
histogram(tout(1,nbi:end),'Normalization','pdf'); hold on;
plot(t(1),0,'o',t_ls(1),0,'s',t_mean(1),0,'^'); hold off;
legend('histogram rwm','true','least squares','mean rwm');
subplot(3,2,3); plot(tout(2,:))
subplot(3,2,4);
histogram(tout(2,nbi:end),'Normalization','pdf'); hold on;
plot(t(2),0,'o',t_ls(2),0,'s',t_mean(2),0,'^'); hold off;
legend('histogram rwm','true','least squares','mean rwm');
subplot(3,2,5); plot(tout(3,:))
subplot(3,2,6);
histogram(tout(3,nbi:end),'Normalization','pdf'); hold on;
plot(t(3),0,'o',t_ls(3),0,'s',t_mean(3),0,'^'); hold off;
legend('histogram rwm','true','least squares','mean rwm');

% Plot IC
nboot   = 1e4;
samp_t  = tout(:,randsample(nbi:end,nboot,'true'));
mod_out = samp_t(1,:).*x+samp_t(2,:);
mod_ic  = quantile(mod_out',[.05 .95]);
y_out   = mod_out + normrnd(0,repmat(samp_t(3,:),numel(x),1));
y_ic    = quantile(y_out',[.05 .95]);
figure
plot(x,f(t,x),'k-'); hold on;
plot(x,y,'ro');
plot(x,f(t_mean,x),'r-');
plot(x,mod_ic','r--');
plot(x,y_ic','r:'); hold off;
