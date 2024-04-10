module CellTrend
using DifferentialEquations, Optimization, LossFunctions, Statistics,
		OptimizationOptimisers, OptimizationOptimJL, Random, Plots,
		Printf, Dates, Serialization, Measures
include("Data.jl")
using .Data
include("/Users/steve/sim/zzOtherLang/julia/modules/MMAColors.jl")
using .MMAColors
export ema, rw, driver_opt, loss, plot_data

# NOTE: The ode has an explicit solution that is easily written down. So
# it might be more efficient to use that solution. However, the solution
# requires numerically evaluating an integral, and so using the solution
# instead of the ode trades numerical analysis of an integral instead of
# an ode. The current code runs so quickly and is suited to purpose, so 
# no need to change.

sigmoidc(x; epsilon=1e-5) = clamp(1 / (1 + exp(-x)), epsilon, 1-epsilon)

# globals for callback
acc_ma = 0
iter = 0

function driver_opt(T=30; maxiters=10, save=true, saveat=0.1, rstate = nothing,
		dir="/Users/steve/Desktop/", learn=0.005, scale=1e1, restart="",
		ρ=0)
	if rstate === nothing
		rstate = copy(Random.default_rng())
		println(rstate)
	end
	copy!(Random.default_rng(), rstate)
	global acc_ma = 0.5
	global iter = 0
	u0 = [1,1]

	nparam = 3 + 0
	if restart !== ""
		d = deserialize(dir * restart)
		p = d.p
	else
		p = find_eq(u0, nparam)
	end
	optf = OptimizationFunction(
			(p,x) -> loss(p,T,u0; saveat=saveat,scale=scale),
			Optimization.AutoForwardDiff())
	prob = OptimizationProblem(optf,p)
	# OptimizationOptimisers.Sophia(η=0.001)
	s = solve(prob, OptimizationOptimisers.Sophia(η=0.001,ρ=ρ), # η=001, ρ=0.04
				maxiters=maxiters, callback=callback)
	d = (p=s.u, loss=s.objective[1], T=T, maxiters=maxiters, rstate=rstate,
			saveat=saveat, u0=u0, learn=learn, scale=scale, ρ=ρ)
	if save
		outfile = save_results(d; dir=dir)
		println("Out file = ", outfile)
	end
	return d
end

function save_results(d; dir="/Users/steve/Desktop/")
	date = Dates.format(Dates.now(), "yyyy-mm-dd-HH-MM-SS");
	serialize(dir * "$date.jls", d);
	return "$date.jls"
end

function find_eq(u0, nparam; maxiters=10000)
	p = 1e0 * rand(nparam) .+ 0.1		# must adjust weighting for equil values
	optf = OptimizationFunction((p,x) -> loss_eq(p,u0))
	prob = OptimizationProblem(optf,p)
	p = solve(prob, NelderMead(), maxiters=maxiters,callback=callback)
	println("Eq loss = ", loss_eq(p,u0))
	return p
end

function ode(u, p, t, v)
	du1 = p[1]*v(t) - p[2]*u[1]
	du2 = p[3]*(p[1]*v(t) - p[2]*u[2])	# different response, same equil
#	du1 = v(t) - p[1]*u[1]
#	du2 = p[2]*(v(t) - p[1]*u[2])	# different response, same equil
	return [du1,du2]
end

# In Optimization.jl, confusing notation. There, u is optimized, and p
# are constant parameters for the OptimizationFunction. Here, p are the
# parameters to be optimized, and u are the ode variables
# Train so that p1 is input at last step, p2 is current input - prev,
# with p1 and p2 both adjust by affine transformation of parameters
function loss(p, T, u0; saveat=0.1, skip=0.1, scale=1e1)
	tspan = (0,T)
	rescale = scale/saveat
	y,t = rw(T/rescale; sigma=0.2, saveat=saveat, low=0.25, high=0.75)
	v = ema_interp(y,rescale*t)
	prob = ODEProblem((u,p,t) -> ode(u, p, t, v), u0, tspan, p)
	sol = solve(prob, Tsit5(), saveat=scale, maxiters=100000)
	# BS3 about 1/3 faster but less accurate
	#sol = solve(prob, BS3(), saveat=scale, maxiters=100000)
	skip = Int(floor(skip*length(sol.t)))
	y_diff = calc_y_diff(y,1)[1+skip:end]
	y_true = calc_y_true(y_diff)
	s1 = sol[1,:][1+skip:end]
	s2 = sol[2,:][1+skip:end]
	sdiff = s1 .- s2
	s = [s1,s2]
	yp = sigmoidc.(sdiff)[1:end-1]
	lm = sum(CrossEntropyLoss(),yp,y_true)
	return lm, yp, y_true, s, y, p, y_diff, skip
end

function loss_eq(p, u0)
	v = x -> 0.5
	du = ode(u0,p,0.0,v)
	return sum(abs2.(du))
end

function callback(state, loss, yp, y_true, s, y, p, y_diff, skip; direct=false)
	global iter += 1
	acc = accuracy(yp,y_true)
	max_acc = prob_pred_next(y)
	pred_pos = sum((yp .> 0.5) .== true) / length(yp)
	global acc_ma = 0.98*acc_ma + 0.02*acc
	if iter % 100 == 0 || direct == true
		@printf("%4d: Loss = %8.2e, accuracy = %5.3f, max_acc = %5.3f, fr_pos = %5.3f\n",
			iter, loss, acc_ma, max_acc, pred_pos)
		pl = plot(layout=(3,1),size=(800,900),legend=:none)
		plot!(y[1+skip:end],subplot=1)
		plot!([s[1],s[2]],subplot=2)
		plot!(2*(yp .- 0.5),subplot=3)
		display(pl)
	end
	return false
end

function callback(state, loss)
	#println("Loss2 = ", loss)
	return false
end

function plot_data(T,d; rstate = nothing, subset=false)
	rstate === nothing ?
		println(copy(Random.default_rng())) :
		copy!(Random.default_rng(), rstate)
	loss_val, yp, y_true, sol, y, p, y_diff, skip = 
		loss(d.p, d.T, d.u0,; saveat=d.saveat, scale=d.scale)
	acc = accuracy(yp,y_true)
	max_acc = prob_pred_next(y)
	pred_pos = sum((yp .> 0.5) .== true) / length(yp)
	@printf("\nLoss = %8.2e, accuracy = %5.3f, max_acc = %5.3f, fr_pos = %5.3f\n\n",
			loss_val, acc, max_acc, pred_pos)
	
	wd = 2
	
	ly = length(y)
	r = skip:(ly-1)
	rs = firstindex(sol[1]):lastindex(sol[1])
	rr = r[begin]:(r[end]-1)
	ryp = 1:length(rr)
	if subset
		steps=10
		lmid = Int(round(ly/2))
		r = (lmid-steps):(lmid+steps)
		rr = r
		rs = r .- skip
		ryp = rs
	end
	yy = y[1+skip:end]
	yy = yy * mean(sol[1]) / mean(yy)
	pl = plot(layout=(3,1),size=(600,800),legend=:none)
	plot!(r,sol[1][rs],color=mma[1],w=3.5,subplot=1)
	plot!(r,yy[rs],color=mma[4],w=1.5,left_margin=0.7cm,subplot=1)
	plot!(r,sol[1][rs],color=mma[1],w=wd,subplot=2)
	plot!(r,sol[2][rs],color=mma[2],w=2,subplot=2)
	plot!(rr,1000*(yp[ryp] .- 0.5),color=mma[1],w=wd,
			bottom_margin=0.5cm,subplot=3)

	annotate!(pl[3],(0.49,-0.18),"Temporal sample points",15)
	annotate!(pl[1],(-0.10,0.5),text("Molecular abundance",12,rotation=90))
	annotate!(pl[2],(-0.10,0.5),text("Molecular abundance",12,rotation=90))
	annotate!(pl[3],(-0.10,0.52),text("Predicted direction",12,rotation=90))
	chrs = 'a':'z'
	for i in 1:3
		annotate!(pl[i],(0.05,0.99),text(@sprintf("(%s)",chrs[i]),10))
	end

	display(pl)
	return pl
end

end # module CellTrend
