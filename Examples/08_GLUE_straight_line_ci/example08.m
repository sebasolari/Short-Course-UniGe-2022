% Generates 100 samples
f = @(tt,x) tt(1).*x+tt(2);
x = linspace(0,10,100)';
t = [.5;5];
y = f(t,x) + normrnd(0,1,numel(x),1);

% Informal likelihood and behavioural limit
L    = @(tt) 1-sum((y-f(tt,x)).^2)/sum((y-mean(y)).^2);
minL = 0.5;

% Sample parameter space and evaluate likelihood
nsim = 1e4;
p    = numel(t);
tt   = zeros(nsim,p);
lob  = [-2 0];
upb  = [2 10];
for id = 1:p
    tt(:,id) = unifrnd(lob(id),upb(id),nsim,1);
end
ttL = zeros(nsim,1);
for id = 1:nsim
    ttL(id) = L(tt(id,:));
end

% Retain only behavioural parameters and reescale likelihood
idbe = find(ttL>minL);
tt   = tt(idbe,:);
ttL  = ttL(idbe);
ttL  = ttL./sum(ttL);

% Plot parameter space
figure
scatter(tt(:,1),tt(:,2),16,ttL,'filled')

% Plot IC
nboot  = 1e4;
samp_t = tt(randsample(numel(ttL),nboot,'true',ttL),:);
y_out  = samp_t(:,1)'.*x+samp_t(:,2)';
y_ic   = quantile(y_out',[.05 .95]);
figure
plot(x,f(t,x),'k-'); hold on;
plot(x,y,'ro');
plot(x,y_ic','r:'); hold off;