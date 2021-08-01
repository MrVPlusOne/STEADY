# this script is supposed to be run inside the module SEDL

using DrWatson
# @quickactivate "ProbPRL"

using StatsPlots
using BenchmarkTools
using Turing
using Optim
using UnPack

# include("../src/SEDL/SEDL.jl")
using .Car1D: data_process, infer_process

"""
Plot the result returned by `Car1D.data_process`.
"""
function plot_result(result)
    @unpack states, times, wall_pos, data = result
    xs = times[1:end-1]
    p_state = plot(xs, states', label=["x" "v"], title="States")
    hline!(p_state, [wall_pos], label="wall_x")
    p_action = plot(xs, data.actions', label=["F"], title="Actions")
    p_obs = plot(xs, data.sensor_readings, label="sensor", title="Observations")
    plot!(p_obs, xs, data.speed_readings, label="speed")
    plot!(p_obs, xs, data.odometry_readings, label="odometry")
    plot(p_state, p_action, p_obs, layout=(3,1), size=(600,600))
end

##
Δt = 0.1
tend = 10.0
times = 0:Δt:tend

##
process_prior = data_process(times; euler = false, drag=0.2, mass=2.0, wall_pos=11.0)
prior_run = process_prior() 
plot_result(prior_run)

##
ill_prior = data_process(times; euler=false, drag=10.0, mass=0.1, wall_pos=11.0)
ill_prior() |> plot_result
@benchmark sample(ill_prior, Prior(), 1000)
@benchmark for _ in 1:100 ill_prior() end
Main.@profview sample(ill_prior, Prior(), 1000)
sample(ill_prior, Prior(), 100)

##
process_infer_slow = data_process(times; euler=false, prior_run.data...)
process_infer_fast = infer_process(times; euler=false, prior_run.data...)
println("Performing MAP estimation...")
optim_options = Optim.Options(x_abstol=1e-3)

println("benchmarking process_infer_slow...")
@benchmark optimize(process_infer_slow, MAP(), optim_options, autodiff=:forwarddiff)
println("benchmarking process_infer_fast...")
@benchmark optimize(process_infer_fast, MAP(), optim_options, autodiff=:forwarddiff)

map_est = @time optimize(process_infer_slow, MAP(), optim_options, autodiff=:forwarddiff)

##
map_params = [k => map_est.values[k] for k in [:drag, :mass, :wall_pos]]
function check_lp(step::Int)
    process_map = data_process(times[1:step]; actions=prior_run.data.actions, map_params...)
    chain_map = sample(process_map, Prior(), 1)
    chain_map[:lp]
end
process_run_map = 
    data_process(times[1:step]; actions=prior_run.data.actions, map_params...)()
plot_result(process_run_map)

##
println("Sampling posterior...")
@benchmark chains = Turing.sample(process_infer_slow, MH(), 10_000, init_theta=map_est)
println("Sampling from process_infer_fast...")
@benchmark chains = Turing.sample(process_infer_fast, MH(), 10_000, init_theta=map_est)
Main.@profview chains = Turing.sample(process_infer_fast, MH(), 10_000, init_theta=map_est)
chains = Turing.sample(process_infer_slow, MH(), 10_000, init_theta=map_est)
summarize(chains)
plot(chains)

##
