using CellTrend

y,t = CellTrend.rw(300);
v = CellTrend.ema_interp(y,t);
r = 80:0.1:100

du = ones(8);
CellTrend.ode!(du, rand(8), rand(30), 1, 4, v)

y = CellTrend.rw(300);
