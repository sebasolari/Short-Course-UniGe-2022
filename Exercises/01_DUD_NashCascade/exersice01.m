% Generates synthetic data
pp  = [zeros(3,1);5;15;3;zeros(20,1);3;25;10;2;zeros(30,1)]';
th = [2;4;.5];
q0  = nashmod(th,pp);
q  = q0 + normrnd(0,.1.*q0);