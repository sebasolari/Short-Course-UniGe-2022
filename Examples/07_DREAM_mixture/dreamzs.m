function out = dreamzs(logl,logp,par)
% Output:
% logl --> log-likelihood function
% logl --> log-prior function
% par  --> parameters of DREAM(ZS)
% out  --> z,pz,r
%
% Input:
% par.d   --> Number of parameter (dimensions)
% par.nc  --> Number of chains
% par.upb --> Upper bound for initial sample of parameters
% par.lob --> Lower bound for initial sample of parameters
%
% Other default parameters:
% par.m0    --> Number of initial samples
% par.kt    --> Thinning rate
% par.delt  -->
% par.ncr   --> Number of crossover probabilities (3)
% par.psnoo --> Snooker update probability (0.1)
% par.pg1   --> Gamma=1 probability (0.2)
% par.c    = 0.1;
% par.cast = 1e-6;

par.m0    = 100.*par.d;
par.kt    = 10;
par.delt  = 1;
par.ncr   = 3;
par.psnoo = .1;
par.pg1   = .2;
par.c     = .1;
par.cast  = 1e-6;
% if par.d<=25
%     xpd = 2000;   % number of chain elements (or samples) per dimension
% elseif par.d<=50
%     xpd = 4000;
% elseif par.d<=100
%     xpd = 10000;
% else
%     xpd = 20000;
% end
xpd = 10000;


niter    = round(par.d*xpd/par.kt); % number of z values to add per chain
niterbin = round(0.1*niter);        % number of burn-in z values per chain

% Initialize output variables
out.z   = zeros(par.m0+(niterbin+niter)*par.nc,par.d);
out.p   = zeros(par.m0+(niterbin+niter)*par.nc,1);
out.r   = NaN(niter,par.d);

% Initialize crossover variables
cr  = (1:par.ncr)'./par.ncr;
pcr = ones(par.ncr,1)./par.ncr;
jumcr = zeros(par.ncr,1);
nidcr = zeros(par.ncr,1);

% Generates and evaluates initial sample
out.z(1:par.m0,:) = rand_m0_z(par,par.m0);
for id = 1:par.m0
    out.p(id)   = logl(out.z(id,:)) + logp(out.z(id,:));
end
idbad = find(isinf(out.p));
nbad  = numel(idbad);
while nbad>0
    out.z(idbad,:) = rand_m0_z(par,nbad);
    for id = 1:nbad
        out.p(idbad(id)) = logl(out.z(idbad(id),:)) + logp(out.z(idbad(id),:));
    end
    idbad = find(isinf(out.p));
    nbad  = numel(idbad);
end

% Defines initial state of chains
[px,idx] = sort(out.p(1:par.m0),'descend');
px = px(1:par.nc);
x  = out.z(idx(1:par.nc),:);

% Run DREAM(ZS) algorithm
m      = par.m0; % matrix z index
burnin = true;   % auxiliary variable to know if still in burn-in
for iditer = 1:niterbin+niter        % For each iteration...
    for idx = 1:par.kt               % ... runs kt updates...
        for idc = 1:par.nc           % ... for each chain.
            % (1) Select which dimensions to update (crossover).
            idcr = randsample(1:par.ncr,1,'true',pcr);
            ucr  = rand(1,par.d);
            dim  = find(ucr<cr(idcr));
            if isempty(dim)
                dim = randsample(par.d,1);
            end
            ndim = numel(dim);
            % (2) Generates new proposal using past states Z (adaptive or
            % snooker).
            zet  = mvnrnd(zeros(1,ndim),par.cast.*ones(1,ndim));
            if rand<1-par.psnoo % (2.a) Adaptive proposal
                delt = randsample(1:par.delt,1); % número de a,b a usar
                gam  = randsample([1 2.38/sqrt(2*delt*ndim)],1,'true',...
                    [par.pg1 1-par.pg1]);
                lam  = unifrnd(-par.c,par.c,1,ndim);
                ab      = randsample(m,2*delt);
                a       = ab(1:delt);
                b       = ab(delt+1:end);
                dz      = sum(out.z(a,:),1)-sum(out.z(b,:),1);
                dx      = zeros(1,par.d);
                dx(dim) = (1+lam).*gam.*dz(dim) + zet;
            else % (2.b) Snooker proposal
                gam  = unifrnd(1.2,2.2);
                uu = zeros(1,par.d);
                while all(uu==0)
                    abc  = randsample(m,3);
                    uu   = out.z(abc(1),:)-x(idc,:);
                end
                uu   = uu./norm(uu);
                pp   = uu'*uu;
                dz   = pp*out.z(abc(2),:)'-pp*out.z(abc(3),:)';
                dx(dim) = gam.*dz(dim)' + zet;
            end
            xnew = x(idc,:) + dx;
            % (3) Metropolis for accepting/rejecting proposal
            pnew = logl(xnew)+logp(xnew);
            pacc = min(1,exp(pnew-px(idc)));
            if rand<=pacc
                x(idc,:) = xnew;
                px(idc)  = pnew;
            else
                dx = zeros(1,par.d);
            end
            % If still in burn-in iterations, updates crossover info
            if burnin
                jumcr(idcr) = jumcr(idcr) + sum((dx./std(out.z(1:m,:))).^2);
                nidcr(idcr) = nidcr(idcr) + 1;
            end
        end
    end
    % If still in burn-in iterations, updates crossover info
    if burnin && all(jumcr>0)
        pcr = jumcr./nidcr;
        pcr = pcr./sum(pcr);
    end
    % Appends new info to Z
    out.z(m+1:m+par.nc,:) = x;
    out.p(m+1:m+par.nc)   = px;
    m                     = m + par.nc;
    % Chwk if still in preallocated burn-in period.
    if iditer > niterbin
        burnin = false;
    end
    % If burn-in finished, estimates R statistic
    if ~burnin
        idr  = (m-round(iditer/2)*par.nc+1):m;
        nidr = numel(idr);
        zaux = reshape(out.z(idr,:)',par.d,par.nc,nidr/par.nc);
        w    = mean(var(zaux,0,3),2);
        b    = var(mean(zaux,3),0,2);
        out.r(iditer,:) = ...
            sqrt( (nidr-1)/nidr + (1+1/par.nc).*b./w )';
    end
end
% Discard first m0 samples (not part of the chains).
out.z = out.z(par.m0+1:end,:);
out.p = out.p(par.m0+1:end);
out.xr = par.nc.*(1:niterbin+niter)';


function m0_z = rand_m0_z(par,m0)
%
% Realiza el muestreo inicial del espacio de parámetros para ejecutar el
% modelo original
%
% par.d --> dimensions
% par.upb
% par.lob
% par.m0

ntry = 1000;

samp_min = unifrnd(0,1,[m0,par.d]);
dist_min = sum(1./pdist(samp_min));
for itry = 1:ntry
    samp_aux = unifrnd(0,1,[par.m0,par.d]);
    dist_aux = sum(1./pdist(samp_aux));
    if dist_aux<dist_min
        samp_min = samp_aux;
        dist_min = dist_aux;
    end
end
m0_z = par.lob + samp_min.*(par.upb - par.lob);


