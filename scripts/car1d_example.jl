# this script is supposed to be run inside the module SEDL

##

using DrWatson
# @quickactivate "ProbPRL"

using StatsPlots
using BenchmarkTools
using Turing, Zygote, ReverseDiff
using Optim
using GalacticOptim
using GalacticOptim: AutoForwardDiff, AutoTracker, AutoZygote, AutoReverseDiff
using UnPack
using TransformVariables

# include("../src/SEDL/SEDL.jl")
using .Car1D: data_process, infer_process, posterior_density

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

process_infer = data_process(times; euler=true, prior_run.data...)
process_infer_fast = infer_process(times; euler=true, prior_run.data...)
optim_options = Optim.Options(x_abstol=1e-3)

##

println("Performing MAP estimation...")
map_est = @time optimize(process_infer_fast, MAP(), optim_options)
# map_est = @time optimize(process_infer, MAP(), optim_options, autodiff=:reversediff)

##
# MAP estimation using handcrafted score
function MAP_infer(times, data; euler, ad = AutoForwardDiff())
    x_trans = as((
        drag=as(Real, 0., 1.), 
        mass=as(Real, 0.5, 5.0), 
        wall_pos=as(Real, 0., 50.),
        states=as(Array, 2, length(times)-1),
    ))
    function f(x, _)
        -posterior_density(transform(x_trans, x), times, data; euler)
    end
    of = OptimizationFunction(f, ad)
    states0 = zeros(2, length(times)-1)
    x0 = (drag=0.1, mass=1.0, wall_pos=49.0, states=states0)
    v0 = inverse(x_trans, x0)
    prob = OptimizationProblem(of, v0)
    sol = solve(prob, LBFGS(), x_abstol=1e-3, progress=true)
    score = -f(sol, ())
    transform(x_trans, sol), score
end

map_result, score = @time MAP_infer(times, prior_run.data, euler=false, ad=AutoReverseDiff())
result_infered = (; map_result.states, times, map_result.wall_pos, prior_run.data)
plot_result(result_infered)

##

prior_run.data

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
Turing.setadbackend(:reversediff)
chains = Turing.sample(process_infer_fast, NUTS(), 200, init_theta=map_est)
summarize(chains)
# plot(chains)

##