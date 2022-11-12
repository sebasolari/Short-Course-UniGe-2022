function q = nashmod(th,pp)
% Parameters
n  = th(1);
k  = th(2);
q0 = th(3);
% Output q
nt  = numel(pp);
t   = (1:nt)';
iuh = (t./k).^(n-1)./(k.*gamma(n)).*exp(-t./k); 
q   = zeros(nt,nt);
for id = 1:nt
    q(id:nt,id) = pp(id).*iuh(1:nt-id+1);
end
q = sum(q,2)+q0;