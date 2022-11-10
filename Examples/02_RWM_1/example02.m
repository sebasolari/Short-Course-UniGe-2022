% Generates 100 samples
f = @(tt,x) tt(1).*x+tt(2);
x = linspace(0,10,100)';
t = [.5;5];
y = f(t,x) + normrnd(0,1,numel(x),1);

% Defines functions
logPi = @(tt) 0;
logL  = @(tt) -numel(y)/2*log( sum( (tt(1).*x+tt(2) - y).^2 ) );

% Run RWM
nsim = 1e5;
tout = rwm([0;0],logPi,logL,nsim);

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
t_ls = regress(y,[x ones(size(x))]);
% Plot MCMC evolution and histograms
figure
subplot(2,2,1); plot(tout(1,:))
subplot(2,2,2);
histogram(tout(1,nbi:end),'Normalization','pdf'); hold on;
plot(t(1),0,'o',t_ls(1),0,'s',t_mean(1),0,'^'); hold off;
legend('histogram rwm','true','least squares','mean rwm');
subplot(2,2,3); plot(tout(2,:))
subplot(2,2,4);
histogram(tout(2,nbi:end),'Normalization','pdf'); hold on;
plot(t(2),0,'o',t_ls(2),0,'s',t_mean(2),0,'^'); hold off;
legend('histogram rwm','true','least squares','mean rwm');