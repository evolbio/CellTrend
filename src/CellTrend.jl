module CellTrend
using DifferentialEquations, Optimization, LossFunctions, Statistics,
		OptimizationOptimisers, OptimizationOptimJL
include("Data.jl")
using .Data
include("/Users/steve/sim/zzOtherLang/julia/modules/MMAColors.jl")
using .MMAColors
export ema, rw

sigmoidc(x; epsilon=1e-5) = clamp(1 / (1 + exp(-x)), epsilon, 1-epsilon)

function driver(T=300; maxiters=10)
	n = 4
	p = find_eq(10000)
	p[32] = 1.0			# set so that u[8] = p[32], and guess is 0.5
	optf = OptimizationFunction((p,x) -> loss(p,n,T),Optimization.AutoForwardDiff())
	#optf = OptimizationFunction((p,x) -> loss(p,n,T))
	prob = OptimizationProblem(optf,p,lb=1e-5*ones(32),ub=100*ones(32))
	prob = OptimizationProblem(optf,p)
	solve(prob, OptimizationOptimisers.Sophia(η=0.01), maxiters=maxiters, callback=callback)
end

function find_eq(maxiters=10)
	n = 4
	p = 10 * rand(32) .+ 0.1
	optf = OptimizationFunction((p,x) -> loss_eq(p,n))
	prob = OptimizationProblem(optf,p)
	solve(prob, NelderMead(), maxiters=maxiters,callback=callback)
end

# requires 30 parameters, length(p) == 30
function ode!(du, u, p, t, n, v)
	@assert n == 4
	u_m = @view u[1:n]			# mRNA
	u_p = @view u[n+1:2n]		# protein
	m_a = @view p[1:n]
	m_d = @view p[n+1:2n]
	p_a = @view p[2n+1:3n]
	p_d = @view p[3n+1:4n]
	
	input = v(t)
	
	β = @view p[4n+1:5n]
	k = p[5n+1]
	b = @view p[5n+2:5n+10]
	
	a1 = b[3]*u_p[1] + b[4]*u_p[2] + b[1]*input
	a2 = b[5]*u_p[1] + b[6]*u_p[2] + b[2]*input
	a3 = b[7]*u_p[1] + b[8]*u_p[2]
	a4 = b[9]*u_p[3]
	
	a = map(x -> x < 0 ? 0 : x^k, [a1,a2,a3,a4])
	
	#a = [a1,a2,a3,a4].^k
	
	f = [a[i]/(β[i]+a[i]) for i in 1:n]
	
	du[1:n] = m_a .* f .- m_d .* u_m			# mRNA level
	du[n+1:2n] = p_a .* u_m .- p_d .* u_p		# protein level
	return nothing
end

# In Optimization.jl, confusing notation. There, u is optimized, and p
# are constant parameters for the OptimizationFunction. Here, p are the
# parameters to be optimized, and u are the ode variables
function loss(p, n, T; saveat=0.1, skip=0.1)
	skip = Int(floor(skip*T/saveat))
	u0 = ones(2*n)
	tspan = (0,T)
	y,t = rw(T; saveat=saveat)
	v = ema_interp(y,t)
	prob = ODEProblem((du,u,p,t) -> ode!(du, u, p, t, n, v), u0, tspan, p)
	sol = solve(prob, Rodas4P(), saveat=saveat, maxiters=100000)
	y_diff = calc_y_diff(y,1)
	y_true = calc_y_true(y_diff)[1+skip:end]
	s = @view sol[2*n,:][1+skip:end-1]
	yp = sigmoidc.(p[31] .* (s .- p[32]))
	return sum(CrossEntropyLoss(),yp,y_true), yp, y_true, sol
end

function loss_eq(p, n)
	u0 = vcat(100*ones(n),1*ones(n))
	#u0 = ones(2*n)
	du = ones(2*n)
	v = x -> 0.5
	ode!(du,u0,p,0.0,n,v)
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
