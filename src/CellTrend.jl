module CellTrend
using DifferentialEquations, Integrals, Plots
include("/Users/steve/sim/zzOtherLang/julia/modules/MMAColors.jl")
using .MMAColors
export ema, rw

ema_steps(v, T; theta=0.2, saveat=0.1) = [ema(v,theta,t) for t in 0:saveat:T]
# diff between future and curr values
calc_y_diff(y, offset) = y[1+offset:end] - y[1:end-offset]
calc_y_true(y_diff) = map(x -> ifelse(x==0.0, Float64(rand(0:1)),
										ifelse(x>0.0,1.0,0.0)), y_diff)

# for input ema, how often does prior diff pred next diff => max success
function prob_pred_next(v, T; theta=0.2)
	steps = ema_steps(v,T; theta=theta)
	y_diff=calc_y_diff(steps,1);
	y_true = calc_y_true(y_diff)
	matches = [y_true[i+1] == y_true[i] ? 1 : 0 for i in 1:(lastindex(y_true)-1)]
	sum(matches) / (length(y_true)-1)
end

function ode!(du, u, p, t, n, v)
	u_m = @view u[1:n]			# mRNA
	u_p = @view u[n+1:2n]		# protein
	m_a = @view p[1:n]
	m_d = @view p[n+1:2n]
	p_a = @view p[2n+1:3n]
	p_d = @view p[3n+1:4n]
	
	input = ema(v, 0.2, t)
	
	β = @view p[4n+1:5n]
	k = p[5n+1]
	b = @view p[5n+2:5n+10]
	
	a1 = b[3]*u_p[1] + b[4]*u_p[2] + b[1]*input
	a2 = b[5]*u_p[1] + b[6]*u_p[2] + b[2]*input
	a3 = b[7]*u_p[1] + b[8]*u_p[2]
	a4 = b[9]*u_p[3]
	
	a = [a1,a2,a3,a4].^k
	
	f = [a[i]/(b[i]+a[i]) for i in 1:n]
	
	du[1:n] = m_a .* f .- m_d .* u_m			# mRNA level
	du[n+1:2n] = p_a .* u_m .- p_d .* u_p		# protein level
end

# f is function f(t,p), where p is not used
function ema(f, theta, t; start_t = 0)
	ff(τ,p) = f(τ,p)*exp((τ-t)/theta)
	prob = IntegralProblem(ff,start_t,t)
	top = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3).u
	gg(τ,p) = exp((τ-t)/theta)
	prob = IntegralProblem(gg,start_t,t)
	bottom = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3).u
	return top/bottom
end

# random walk
function rw(T; sigma = 0.2, dt = 0.01, norm_rng = true)
    prob = SDEProblem((u, p, t) -> 0, (u, p, t) -> sigma, 0, (0, T))
    sol = solve(prob, EM(), dt = dt)
    return norm_rng ? normalize(sol) : (t,p) -> sol(t)
end

function normalize(sol; low = 0, high = 1)
	mx = maximum(sol)
	mn = minimum(sol)
	return (t,p) -> ((high - low) / (mx - mn))*sol(t) + (low - mn) / (mx - mn)
end

function plot_rw(T; theta=0.2, r=nothing)
	v = CellTrend.rw(T);
	if r === nothing
		r = Int(floor(T/4)):0.1:Int(floor(T/3))
	end
	plot([v(x,0) for x in r],legend=:none, color=mma[1],w=2)
	plot!([ema(v,theta,t) for t in r], color=mma[2],w=2)
end

########################## Not used ##################################

function normalize2(sol; low = 0, high = 1)
	mx = maximum(sol)
	mn = minimum(sol)
	println(mx)
	println(mn)
	f = OptimizationFunction((t,p) -> sol(t))
	prob = OptimizationProblem(f, [maximum(sol.t)/2], [maximum(sol.t)/2];
			lcons=[0], ucons=[300])
	mn = solve(prob, CMAEvolutionStrategyOpt(), maxiters=10000, abstol=1e-24)
	println(mn.u)
	return mn
	#return (t,p) -> ((high - low) / (mx - mn))*sol(t) + (low - mn) / (mx - mn)
end

end # module CellTrend
