using CellTrend, Serialization

y,t = CellTrend.rw(300);
v = CellTrend.ema_interp(y,t);
r = 80:0.1:100

CellTrend.ode(rand(3), rand(7), 1, v)

y = CellTrend.rw(300);

######################

d = driver_opt(5000;maxiters=20, save=false, saveat=0.1,
		dir="/Users/steve/Desktop/", learn=0.005);

d=driver_opt(30000; maxiters=10000, save=false, scale=300);

######################

using CellTrend, Serialization, Random

rstate = Random.Xoshiro(0x69a7dd129e7f60d5, 0x1e7da8f49b6237ba, 0x8317961216fc7d81, 0x2fc455ed74348caf, 0x7db9fb8192742ed2);
d=driver_opt(30000; maxiters=10000, save=true, scale=300, rstate=rstate);

# Read results back in and test
dir = "/Users/steve/Desktop/";
d = deserialize(dir * "2024-04-05-13-08-49.jls");
# fields(d)


loss_val, yp, y_true, sol, y, p, n, y_diff = 
	loss(d.p, d.n, d.T, d.u0, d.tf, d.st, d.re; saveat=d.saveat, scale=d.scale);
CellTrend.callback(nothing, loss_val, yp, y_true, sol, y, p, n, y_diff;
		direct=true);

# create plot for publication

