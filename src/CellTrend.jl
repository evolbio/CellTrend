module CellTrend
using DifferentialEquations, Optimization, LossFunctions, Statistics,
		OptimizationOptimisers, OptimizationOptimJL, Lux, Random
include("Data.jl")
using .Data
include("/Users/steve/sim/zzOtherLang/julia/modules/MMAColors.jl")
using .MMAColors
export ema, rw, driver_opt

sigmoidc(x; epsilon=1e-5) = clamp(1 / (1 + exp(-x)), epsilon, 1-epsilon)

function driver_opt(T=300; maxiters=10)
	n = 4
	tf, pp, st, re = make_activation()
	nparam = 4*n + length(pp) + 2		# 2 is for location and scale of prediction
	p = find_eq(nparam, 10000, tf, st, re)
	p[end] = 1.0			# set so that u[8] = p[32], and guess is 0.5
	optf = OptimizationFunction((p,x) -> loss(p,n,T,tf,st,re),
				Optimization.AutoForwardDiff())
	prob = OptimizationProblem(optf,p)
	solve(prob, OptimizationOptimisers.Sophia(η=0.1), maxiters=maxiters, callback=callback)
end

function find_eq(nparam, maxiters, tf, st, re)
	n = 4
	p = 1e0 * rand(nparam) .+ 0.1		# must adjust weighting for equil values
	optf = OptimizationFunction((p,x) -> loss_eq(p,n,tf,st,re))
	prob = OptimizationProblem(optf,p)
	solve(prob, NelderMead(), maxiters=maxiters,callback=callback)
end

function make_activation()
	f1 = Chain(Dense(3 => 2, mish), Dense(2 => 1))
	f2 = Chain(Dense(3 => 2, mish), Dense(2 => 1))
	f3 = Chain(Dense(2 => 2, mish), Dense(2 => 1))
	f4 = Chain(Dense(1 => 1, identity))
	f = Parallel(nothing, f1, f2, f3, f4)
	ps, st = Lux.setup(Random.default_rng(),f)
	pp, re = destructure(ps)
	return f, pp, st, re
end

nn_input(pr,input) =
	([pr[1],pr[2],input],[pr[1],pr[2],input],pr[1:2],[pr[3]])

# requires 51 parameters, length(p) == 16 + 33 + 2 = 51
function ode!(du, u, p, t, n, v, tf, st, re)
	@assert n == 4
	u_m = @view u[1:n]			# mRNA
	u_p = @view u[n+1:2n]		# protein
	m_a = @view p[1:n]
	m_d = @view p[n+1:2n]
	p_a = @view p[2n+1:3n]
	p_d = @view p[3n+1:4n]
	p_tf = @view p[4n+1:end-2]
	
	input = v(t)
		
	f = [tf(nn_input(u_p,input), re(p_tf), st)[1][i][1] for i in 1:n]
	
	du[1:n] = m_a .* f .- m_d .* u_m			# mRNA level
	du[n+1:2n] = p_a .* u_m .- p_d .* u_p		# protein level
	return nothing
end

# In Optimization.jl, confusing notation. There, u is optimized, and p
# are constant parameters for the OptimizationFunction. Here, p are the
# parameters to be optimized, and u are the ode variables
function loss(p, n, T, tf, st, re; saveat=0.1, skip=0.1)
	skip = Int(floor(skip*T/saveat))
	u0 = ones(2*n)
	tspan = (0,T)
	y,t = rw(T; saveat=saveat)
	v = ema_interp(y,t)
	prob = ODEProblem((du,u,p,t) -> ode!(du, u, p, t, n, v, tf, st, re), u0, tspan, p)
	sol = solve(prob, Rodas4P(), saveat=saveat, maxiters=100000)
	y_diff = calc_y_diff(y,1)
	y_true = calc_y_true(y_diff)[1+skip:end]
	s = @view sol[2*n,:][1+skip:end-1]
	yp = sigmoidc.(p[end-1] .* (s .- p[end]))
	return sum(CrossEntropyLoss(),yp,y_true), yp, y_true, sol
end

function loss_eq(p, n, tf, st, re)
	u0 = vcat(1e3*ones(n),1e5*ones(n))
	du = ones(2*n)
	v = x -> 0.5
	ode!(du,u0,p,0.0,n,v,tf,st,re)
	return sum(abs2.(du))
end

function callback(state, loss, yp, y_true, sol)
	println("Loss = ", loss)
	return false
end

function callback(state, loss)
	println("Loss2 = ", loss)
	return false
end

end # module CellTrend
