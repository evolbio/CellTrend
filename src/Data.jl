module Data
using DifferentialEquations, DataInterpolations, Plots
export ema, ema!, rw, plot_rw, calc_y_diff, calc_y_true, prob_pred_next,
	ema_interp, accuracy

# Use discrete saveat points for random walk, then discrete ema,
# then linear interpolation of discrete ema to get continuous inputs
# for ode.

# diff between future and curr values
calc_y_diff(y, offset) = y[1+offset:end] - y[1:end-offset]
calc_y_true(y_diff) = map(x -> ifelse(x==0.0, Float64(rand(0:1)),
										ifelse(x>0.0,1.0,0.0)), y_diff)
matches(y_pred, y_true) = sum((y_pred .> 0.5) .== y_true)
accuracy(y_pred, y_true) = matches(y_pred, y_true) / length(y_pred)
ema!(data, alpha) = ema(data, alpha; in_place = true)
ema_interp(y,t) = LinearInterpolation(y,t)

# for input ema, how often does prior diff pred next diff => max success
function prob_pred_next(y)
	y_diff=calc_y_diff(y,1);
	y_true = calc_y_true(y_diff)
	matches = [y_true[i+1] == y_true[i] ? 1 : 0 for i in 1:(lastindex(y_true)-1)]
	sum(matches) / (length(y_true)-1)
end

function ema(data, alpha; in_place = false)
    y = in_place ? data : deepcopy(data)
    for i = 2:lastindex(y)
        y[i] = alpha * y[i] + (1 - alpha) * y[i-1]
    end
    return y
end

# random walk
function rw(T; sigma = 0.2, saveat = 0.1, alpha = 0.2, norm_rng = true)
    prob = SDEProblem((u, p, t) -> 0, (u, p, t) -> sigma, 0, (0, T))
    sol = solve(prob, EM(), dt = 0.01; saveat = saveat)
    u = norm_rng ? normalize!(sol.u) : sol.u
    return alpha == 1 ? u : ema!(u, alpha), sol.t
end

function normalize!(x; low = 0, high = 1)
    x .= ((high - low) / (maximum(x) - minimum(x)) .* x)
    x .= x .+ (low - minimum(x))
end

function plot_rw(T; theta=0.2, r=nothing)
	y,_ = rw(T;alpha=1);
	if r === nothing r = 1:lastindex(y) end
	plot([y[r]],legend=:none, color=mma[1],w=2)
	plot!(ema!(y[r],0.2), color=mma[2],w=2)
end

end # module Data
