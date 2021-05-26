xmesh = linspace(0,1,100);
solinit = bvpinit(xmesh, @guess);
sol = bvp4c(@bvpfcn, @bcfcn, solinit);
plot(sol.x, sol.y, '-o')

function dydx = bvpfcn(x,y) % equation to solve
dydx = zeros(2,1);
dydx = [y(2)
       3/2*y(1)];
end
%--------------------------------
function res = bcfcn(ya,yb) % boundary conditions
res = [ya(1)-4
       yb(1)-1];
end
%--------------------------------
function g = guess(x) % initial guess for y and y'
g = [sin(x)
     cos(x)];
end