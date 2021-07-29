# this script is supposed to be run inside the module SEDL

using DrWatson
# @quickactivate "ProbPRL"

using StatsPlots
using BenchmarkTools
using Turing
using Optim
using UnPack

# include("../src/SEDL/SEDL.jl")
using .Examples.Car1D

"""
Plot the result returned by `Car1D.data_process`.
"""
function plot_result(result)
    @unpack states, wall_pos, data = result
    xs = times[1:end-1]
    p_state = plot(xs, states', label=["x" "v"], title="States")
    hline!(p_state, [wall_pos], label="wall_x")
    p_action = plot(xs, data.actions', label=["F"], title="Actions")
    plot(p_state, p_action, layout=(2,1), size=(600,600))
end

Δt = 0.1
tend = 10.0
times = 0:Δt:tend

##
process_prior = Car1D.data_process(times; drag=0.2, mass=2.0)
prior_run = process_prior() 
plot_result(prior_run)

##

##
process_infer = Car1D.data_process(times; prior_run.data...)
println("Performing MAP estimation...")
map_est = @time optimize(process_infer, MAP(), autodiff=:forwarddiff)

##
map_params = [k => map_est.values[k] for k in [:drag, :mass, :wall_pos]]
process_map = Car1D.data_process(times; actions=prior_run.data.actions, map_params...)
process_run_map = process_map()
chain_map = sample(process_map, Prior(), 1)
chain_map[:lp]
propertynames(chain_map)
plot_result(process_run_map)

##


chains = Turing.sample(process_infer, NUTS(), 500)
summarize(chains)

@doc init