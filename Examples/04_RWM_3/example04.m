% Generates 200 samples from a truncated Normal
t = [.5;1];
x = normrnd(t(1),t(2),200,1);
while any(x<0)
    id    = find(x<0);
    x(id) = normrnd(t(1),t(2),numel(id),1);
end

% Defines functions
logPi = @(tt) 0;
logL  = @(tt) -numel(x)*log(1-normcdf(0,tt(1),tt(2))) ...
    -numel(x)*log(tt(2)) - .5*sum( ( (x-tt(1))./tt(2) ).^2 );

% Run RWM
nsim = 1e5;
[tout,lout] = rwm([mean(x);std(x)],logPi,logL,nsim);

% Burnin data (estimated)
nbi = round(0.2*nsim);
% Mean value of the parameters
t_mean  = mean(tout(:,nbi:end),2);
[~,idm] = max(lout);
t_maxl  = tout(:,idm);

% Plot results
figure
histogram(x,'Normalization','pdf'); hold on;
plot(0:.01:5,normpdf(0:.01:5,t(1),t(2))./...
    (1-normcdf(0,t(1),t(2))),'k-');
plot(0:.01:5,normpdf(0:.01:5,t_maxl(1),t_maxl(2))./...
    (1-normcdf(0,t_maxl(1),t_maxl(2))),'r-');
plot(0:.01:5,normpdf(0:.01:5,t_mean(1),t_mean(2))./...
    (1-normcdf(0,t_mean(1),t_mean(2))),'r--');
plot(0:.01:5,normpdf(0:.01:5,mean(x),std(x)),'r:');

% Plot MCMC evolution and histograms
figure
subplot(2,2,1); plot(tout(1,:))
subplot(2,2,2);
histogram(tout(1,nbi:end),'Normalization','pdf'); hold on;
plot(t(1),0,'o',t_mean(1),0,'^',t_maxl(1),0,'s'); hold off;
legend('histogram rwm','true','mean rwm','max logL');
subplot(2,2,3); plot(tout(2,:))
subplot(2,2,4);
histogram(tout(2,nbi:end),'Normalization','pdf'); hold on;
plot(t(2),0,'o',t_mean(2),0,'^',t_maxl(2),0,'s'); hold off;
legend('histogram rwm','true','mean rwm','max logL');

% Plot scatter of parameters with likelihood
figure
scatter(tout(1,nbi:end),tout(2,nbi:end),16,lout(nbi:end),'filled')
hold on;
plot(t(1),t(2),'o',t_mean(1),t_mean(2),'^',t_maxl(1),t_maxl(2),'s');
legend('MCMC sample','true','mean rwm','max logL');
hold off;
