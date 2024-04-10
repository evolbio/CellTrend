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

using CellTrend, Serialization, Random, Plots

rstate = Random.Xoshiro(0x69a7dd129e7f60d5, 0x1e7da8f49b6237ba, 0x8317961216fc7d81, 0x2fc455ed74348caf, 0x7db9fb8192742ed2);
d=driver_opt(30000; maxiters=10000, save=true, scale=300, rstate=rstate);

# Read results back in and test
dir = "/Users/steve/Desktop/";
d = deserialize(dir * "2024-04-05-13-08-49.jls");
# fields(d)

# Saved run for generating publication analysis and graphics

dir = "/Users/steve/sim/zzOtherLang/julia/projects/Circuits/" * 
			"01_ML_Analogy/CellTrend/output/";
d = deserialize(dir * "SavedRun_2D.jls");

# use rstate = nothing to test different random seeds
pl = plot_data(d.T, d; rstate=d.rstate, subset=false)
savefig(pl,"/Users/steve/Desktop/cellTrend.pdf");

# publication figure
rstate = Xoshiro(0x0ece70a30236ed33, 0x2e815a5584faccba, 0xcba5412ef2c6d28f, 0x6a274bff0a3e0757, 0xaeeda13e3d25b7d6);
pl = plot_data(d.T, d; rstate=rstate, subset=true)
savefig(pl,"/Users/steve/Desktop/cellTrend.pdf");

# Run optimization with ρ=0.04 or something greater than zero to get parameters
# that increase distance between x and y, with y slower than when accuracy only
# optimized, but increased distance and robustness to perturbation of difference
# means a decline in accuracy.

# start by reading in results from ρ=0.00 solution, which emphasizes accuracy
dir = "/Users/steve/sim/zzOtherLang/julia/projects/Circuits/" * 
			"01_ML_Analogy/CellTrend/output/";
d = deserialize(dir * "SavedRun_2D.jls");

# optimize with ρ=0.04
d=driver_opt(d.T; maxiters=d.maxiters, save=true, scale=d.scale, rstate=d.rstate, ρ=0.04);

# Saved run for generating publication analysis and graphics

dir = "/Users/steve/sim/zzOtherLang/julia/projects/Circuits/" * 
			"01_ML_Analogy/CellTrend/output/";
d = deserialize(dir * "SavedRun_2D_rho4.jls");

# publication figure
rstate = Xoshiro(0x0ece70a30236ed33, 0x2e815a5584faccba, 0xcba5412ef2c6d28f, 0x6a274bff0a3e0757, 0xaeeda13e3d25b7d6);
pl = plot_data(d.T, d; rstate=rstate, subset=true)
savefig(pl,"/Users/steve/Desktop/cellTrend.pdf");

###################################

# testing
loss_val, yp, y_true, sol, y, p, y_diff, skip = 
		loss(d.p, d.T, d.u0,; saveat=d.saveat, scale=d.scale);
CellTrend.callback(nothing, loss, yp, y_true, sol, y, p, y_diff, skip; direct=false);

