function tout = dud(y,f,t0,dt0)
% Inputs
% y   -> set of observations
% f   -> funtion to fit
% t0  -> starting point in parameter space
% dt0 -> initial perturbation for each parameter
%
% Outputs
% tout -> optimum parameters

% Tuneable parameters
maxiter = 100;
mmax = 5;
qtol = 1e-5;
%%%%%%%%%%%%%%%%%%%%

% DUD algorithm
% Initialize variables
p  = numel(t0);
n  = numel(y);
tt = zeros(p,p+1);
fi = zeros(n,p+1);
% Initial set of points in the parameters space
tt(:,1) = t0;
fi(:,1) = f(tt(:,1));
for id = 1:p
    tt(:,id+1)  = tt(:,1);
    tt(id,id+1) = tt(id,1)+dt0(id);
    fi(:,id+1)  = f(tt(:,id+1));
end
qi      = sum((y-fi).^2)';
[qi,id] = sort(qi,'descend');
fi      = fi(:,id);
tt      = tt(:,id);
qi      = qi(id);
% Star iterations
iter  = 0;
evals = p+1;
while iter<maxiter
    iter  = iter+1;
    evals = evals+1;
    % Generates new proposal
    df    = fi(:,1:p)-fi(:,p+1);
    alf   = (df'*df)\(df'*(y-fi(:,p+1)));
    dtt   = tt(:,1:p)-tt(:,p+1);
    im    = 0;
    ttn   = tt(:,p+1) + dtt*alf;
    fn    = f(ttn);
    qn    = sum((y-fn).^2);
    % Check for adequacy of new proposal and correct if required
    while qn>qi(end) && im<mmax
        evals = evals+1;
        im    = im+1;
        d     = -(-.5)^im;
        ttn   = tt(:,p+1) + d*dtt*alf;
        fn    = f(ttn);
        qn    = sum((y-fn).^2);
    end
    % Updates current set of points in the parameters space
    tt = [tt(:,2:end) ttn];
    fi = [fi(:,2:end) fn];
    qi = [qi(2:end);qn];
    % Check convergence
    if abs(qn-qi(end-1))/qi(end-1)<qtol
        tout = ttn;
        disp('DUD converged')
        fprintf('in %d iterations\n',iter)
        fprintf('using %d function evaluations\n',evals)
        disp('to this set of parameters:')
        disp('    tout')
        disp('    -----')
        disp(tout)
        return
    end
end
disp('DUD did not converged')