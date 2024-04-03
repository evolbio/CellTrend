using CellTrend, Serialization

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

d = driver_opt(5000;maxiters=5000,scale=100);

######################

# Overall approach
# 
# -- Use NN tf function to find p1, p2 matches to input-1, input
# -- Set parameters for those TFs
# -- Consider replacing NN with direct diff eq for p1, p2
# -- Find match of p3 to p1-p2 diff, use f3 tf function as k*sigmoid(p1-p2)
# -- Refine match by using CrossEntropy loss to predict next direction of change
# 	instead of p1-p2 diff, or do this instead of prior step

using CellTrend, Serialization

# TEST OTHER ODE SOLVER PERFORMANCE: 
#		https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/

# -- Use NN tf function to find p1, p2 matches to input-1, input
d = driver_opt(10000;maxiters=3000,scale=100,loss_type="12");


# Read results back in and test
dir = "/Users/steve/Desktop/";
d = deserialize(dir * "2024-04-03-10-14-13.jls");
# fields(d)
loss_val, yp, y_true, sol, y, p, n, y_diff = 
	loss(d.p, d.n, d.T, d.u0, d.tf, d.st, d.re; saveat=d.saveat, scale=d.scale);
CellTrend.callback(nothing, loss_val, yp, y_true, sol, y, p, n, y_diff;
		direct=true);

# Optimize p3 as predictor of direction of change
d = driver_opt(10000;maxiters=1000,scale=100,loss_type="all",
				restart="2024-04-03-11-07-49.jls");

