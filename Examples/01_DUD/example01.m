% Generates 100 samples
x = linspace(0,10,100)';
t = [.5;5];
y = t(1).*x+t(2) + normrnd(0,1,numel(x),1);

% Defines function
f = @(tt) tt(1).*x + tt(2);

% Run DUD
tout = dud(y,f,[0;0],[.1;.1]);

% Plot results
figure
plot(x,f(t),'k-'); hold on;
plot(x,y,'ro');
plot(x,f(tout),'r--'); hold off;