module Analysis
using Plots, Printf, Random, CellTrend, Statistics, Measures
export plot_data, plot_err_distn

function prior_next_match(T,d; yy=nothing, prnt=true)
	if yy === nothing
		rescale = d.scale/d.saveat
		yy,_ = rw(T/rescale; sigma=0.2, saveat=d.saveat, low=0.25, high=0.75)
	end
	tt=Vector{Int}(undef,length(yy)-2);
	for i in eachindex(tt)
		tt[i] = ifelse((yy[i+1] > yy[i]) == (yy[i+2] > yy[i+1]), 1, 0)
	end
	mtch = sum(tt)/length(tt)
	if prnt @printf("prior-next match = %5.3f\n", mtch) end
	return mtch
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
	annotate!(pl[1],(-0.10,0.5),text("State value",12,rotation=90))
	annotate!(pl[2],(-0.10,0.5),text("State value",12,rotation=90))
	annotate!(pl[3],(-0.10,0.52),text("Predicted direction",12,rotation=90))
	chrs = 'a':'z'
	for i in 1:3
		annotate!(pl[i],(0.05,0.99),text(@sprintf("(%s)",chrs[i]),10))
	end

	display(pl)
	return pl
end

function plot_err_distn(T,d; n=100, skip=0.1, rstate = nothing, match_test=2000)
	@assert 0.02*n - floor(0.02*n) == 0
	ytk = Int(0.02*n)
	rstate === nothing ?
		println(copy(Random.default_rng())) :
		copy!(Random.default_rng(), rstate)
	err = zeros(n);
	sk = skip
	for i in 1:n
		_, yp, _, _, yy, _, yd, _ =
				loss(d.p,d.T,d.u0;saveat=d.saveat, skip=sk, scale=d.scale);
		skip = Int(floor(sk*length(yy)))
		yy = @view yy[1+skip:end]
		yp = @view yp[1+skip:end]
		yd = @view yd[1+skip:end]
		err[i] = prior_next_match(T,d; yy=yy, prnt=false) - accuracy(yp,calc_y_true(yd))
	end
	pl = histogram(err,legend=:none, color=mma[1],
		xlabel="Deviation from maximum accuracy", ylabel="Frequency",
		yticks=(ytk:ytk:4*ytk,string.(0.02:0.02:0.08)))
	max_acc = prior_next_match(match_test,d)
	annotate!((0.9,0.85),text(@sprintf("%11s = %5.3f", "max accuracy", max_acc),11, :right))
	annotate!((0.9,0.76),text(@sprintf("%11s = %5.3f", "median", median(err)),11, :right))
	annotate!((0.9,0.67),text(@sprintf("%11s = %5.3f", "mean", mean(err)),11, :right))
	annotate!((0.9,0.58),text(@sprintf("%11s = %5.3f", "standard dev", std(err)),11, :right))
	display(pl)
	@printf("median = %7.2e, mean = %7.2e, sd = %7.2e\n", median(err), mean(err), std(err))
	return pl
end

end # module Analysis

