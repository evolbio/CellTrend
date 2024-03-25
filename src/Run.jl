using CellTrend

v = CellTrend.rw(300);
r = 80:0.1:100
plot([ema(v,0.2,t) for t in r],legend=:none)
plot!([v(x,0) for x in r])

CellTrend.ode!(du, rand(8), rand(30), 1, 4, v)

y = CellTrend.rw(300);
