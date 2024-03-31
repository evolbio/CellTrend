using CellTrend

y,t = CellTrend.rw(300);
v = CellTrend.ema_interp(y,t);
r = 80:0.1:100

du = ones(8);
CellTrend.ode!(du, rand(8), rand(30), 1, 4, v)

y = CellTrend.rw(300);

######################

d = driver_opt(5000;maxiters=20, save=false, saveat=0.1, n=4,
		dir="/Users/steve/Desktop/", learn=0.005);
r=rand(d.n);
CellTrend.tf_out(d.tf,r,7,d.re,d.p[17:49],d.st,d.n)

