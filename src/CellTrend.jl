module CellTrend
using DifferentialEquations, Optimization, LossFunctions, Statistics,
		OptimizationOptimisers, OptimizationOptimJL, Lux, Random, Plots,
		Printf, Dates
include("Data.jl")
using .Data
include("/Users/steve/sim/zzOtherLang/julia/modules/MMAColors.jl")
using .MMAColors
export ema, rw, driver_opt

sigmoidc(x; epsilon=1e-5) = clamp(1 / (1 + exp(-x)), epsilon, 1-epsilon)

# globals for callback
acc_ma = 0
iter = 0			

function driver_opt(T=30; maxiters=10, save=false, saveat=0.1, n=4,
		dir="/Users/steve/Desktop/", learn=0.005, scale=1e1)
	global acc_ma = 0.5
	global iter = 0
	u0 = vcat(1e0*ones(n),1e0*ones(n))

	tf, pp, st, re = make_activation()
	nparam = 4*n + length(pp) + 6		# 6 is for location and scale of prediction
	p = find_eq(u0, nparam, tf, st, re)
	p[end] = u0[2*n]			# set so that u[8] = p[end]
	optf = OptimizationFunction(
				(p,x) -> loss(p,n,T,u0,tf,st,re; saveat=saveat,scale=scale),
				Optimization.AutoForwardDiff())
	prob = OptimizationProblem(optf,p)
	# OptimizationOptimisers.Sophia(η=0.001)
	s = solve(prob, OptimizationOptimisers.Sophia(η=0.005), # η=
					maxiters=maxiters, callback=callback)
	d = (p=s.u, loss=s.objective[1], tf=tf, re=re, st=st, n=n, T=T, maxiters=maxiters,
			saveat=saveat, u0=u0, learn=learn, scale=scale)
	if save
		date = Dates.format(Dates.now(), "yyyy-mm-dd-HH-MM-SS");
		serialize(dir * "$date.jls", d);
	end
	return d
end

function find_eq(u0, nparam, tf, st, re; maxiters=10000)
	n = 4
	p = 1e0 * rand(nparam) .+ 0.1		# must adjust weighting for equil values
	optf = OptimizationFunction((p,x) -> loss_eq(p,n,u0,tf,st,re))
	prob = OptimizationProblem(optf,p)
	p = solve(prob, NelderMead(), maxiters=maxiters,callback=callback)
	println("Eq loss = ", loss_eq(p,n,u0,tf,st,re))
	return p
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
tf_out(tf,u_p,input,re,p_tf,st,n) = 
	[tf(nn_input(u_p,input), re(p_tf), st)[1][i][1] for i in 1:n]

# requires 49+x parameters, length(p) == 16 + 33 + x, where x is number used in loss
function ode(u, p, t, n, v, tf, st, re)
	@assert n == 4
	u_m = @view u[1:n]			# mRNA
	u_p = @view u[n+1:2n]		# protein
	m_a = @view p[1:n]
	m_d = @view p[n+1:2n]
	p_a = @view p[2n+1:3n]
	p_d = @view p[3n+1:4n]
	p_tf = @view p[4n+1:end-6]
	
	input = v(t)
	f = tf_out(tf,u_p,input,re,p_tf,st,n)
	
	du_m = m_a .* f .- m_d .* u_m		# mRNA level
	du_p = p_a .* u_m .- p_d .* u_p		# protein level
	return vcat(du_m, du_p)
end

# In Optimization.jl, confusing notation. There, u is optimized, and p
# are constant parameters for the OptimizationFunction. Here, p are the
# parameters to be optimized, and u are the ode variables
# Train so that p1 is input at last step, p2 is current input - prev,
# with p1 and p2 both adjust by affine transformation of parameters
function loss(p, n, T, u0, tf, st, re; saveat=0.1, skip=0.1, scale=1e1)
	tspan = (0,T)
	rescale = scale/saveat
	y,t = rw(T/rescale; sigma=0.2, saveat=saveat, low=0.25, high=0.75)
	v = ema_interp(y,rescale*t)
	y = v.(0:T)
	prob = ODEProblem((u,p,t) -> ode(u, p, t, n, v, tf, st, re), u0, tspan, p)
	sol = solve(prob, Tsit5(), saveat=1.0, maxiters=100000)
	skip = Int(floor(skip*length(sol.t)))
	y_diff = calc_y_diff(y,1)[1+skip:end]
	y_true = calc_y_true(y_diff)
	s1 = p[end-1]*(sol[n+1,:][1+skip:end] .- p[end])
	l1 = sum(abs2.(s1 .- y[skip:end-1]))
	#s2 = p[end-3]*(sol[n+2,:][2+skip:end] .- p[end-2])
	#l2 = sum(abs2.(s2 .- y_diff))	# check time matching
	s2 = p[end-3]*(sol[n+2,:][1+skip:end] .- p[end-2])
	l2 = sum(abs2.(s2 .- y[1+skip:end]))
	s3 = p[end-5]*(sol[n+3,:][1+skip:end-1] .- p[end-4])
	yp = sigmoidc.(s3)
	lm = sum(CrossEntropyLoss(),yp,y_true)
	return l1+l2+lm, yp, y_true, sol, y, p
end

function loss_eq(p, n, u0, tf, st, re)
	v = x -> 0.5
	du = ode(u0,p,0.0,n,v,tf,st,re)
	return sum(abs2.(du))
end

function callback(state, loss, yp, y_true, sol, y, p)
	global iter += 1
	acc = accuracy(yp,y_true)
	pred_pos = sum((yp .> 0.5) .== true) / length(yp)
	global acc_ma = 0.95*acc_ma + 0.05*acc
	if iter % 10 == 0
		@printf("%4d: Loss = %8.2f, accuracy = %5.3f, fr_pos = %5.3f\n",
			iter, loss, acc_ma, pred_pos)
		pl = plot(layout=(5,2),size=(800,750),legend=:none)
		panels = vcat(1:2:7,2:2:8)
		skip = Int(floor(0.1*length(y)))
		for i in 1:8
			plot!(sol[i,1+skip:end],subplot=panels[i])
		end
		plot!(y[skip:end-1], subplot=9,color=mma[1])
		plot!(p[end-1]*(sol[5,1+skip:end] .- p[end]), subplot=9,color=mma[2])
		plot!(y[1+skip:end], subplot=10,color=mma[1])
		plot!(p[end-3]*(sol[6,1+skip:end] .- p[end-2]), subplot=10,color=mma[2])
		display(pl)
	end
	return false
end

function callback(state, loss)
	#println("Loss2 = ", loss)
	return false
end

end # module CellTrend
