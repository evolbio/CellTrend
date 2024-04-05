module CellTrend
using DifferentialEquations, Optimization, LossFunctions, Statistics,
		OptimizationOptimisers, OptimizationOptimJL, Random, Plots,
		Printf, Dates, Serialization
include("Data.jl")
using .Data
include("/Users/steve/sim/zzOtherLang/julia/modules/MMAColors.jl")
using .MMAColors
export ema, rw, driver_opt, loss

sigmoidc(x; epsilon=1e-5) = clamp(1 / (1 + exp(-x)), epsilon, 1-epsilon)

# globals for callback
acc_ma = 0
iter = 0			

function driver_opt(T=30; maxiters=10, save=true, saveat=0.1, n=3,
		dir="/Users/steve/Desktop/", learn=0.005, scale=1e1, restart="")
	global acc_ma = 0.5
	global iter = 0
	u0 = [1,1,0.5]

	nparam = 6 + 1
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
	s = solve(prob, OptimizationOptimisers.Sophia(η=0.005), # η=
					maxiters=maxiters, callback=callback)
	d = (p=s.u, loss=s.objective[1], T=T, maxiters=maxiters,
			saveat=saveat, u0=u0, learn=learn, scale=scale)
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
	du2 = p[3]*(p[1]*v(t) - p[2]*u[2])	# faster response, same equil
	du3 = p[4] + p[5]*(u[2] - u[1]) - p[6]*u[3]
	return [du1,du2,du3]
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
	s3 = p[end]*(sol[3,:][1+skip:end] .- p[4]/p[6])
	s = [s1,s2,s3]
	yp = sigmoidc.(s3)[1:end-1]
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
		plot!(s[3],subplot=3)
		display(pl)
	end
	return false
end

function callback(state, loss)
	#println("Loss2 = ", loss)
	return false
end

end # module CellTrend
