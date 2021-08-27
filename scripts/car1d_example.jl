# this script needs to be run inside the module SEDL

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
using StatsBase
using ProgressLogging
using DataFrames

using .Car1D_old: data_process, posterior_density

"""
Plot the result returned by `Car1D.data_process`.
"""
function plot_result((; states, times, wall_pos, data), name::String)
    xs = times[1:end-1]
    p_state = plot(xs, states', label=["x" "v"], title="States ($name)")
    hline!(p_state, [wall_pos], label="wall_x")
    p_action = plot(xs, data.actions', label=["F"], title="Actions ($name)")
    p_obs = plot(xs, data.sensor_readings, label="sensor", title="Observations ($name)")
    plot!(p_obs, xs, data.speed_readings, label="speed")
    plot!(p_obs, xs, data.odometry_readings, label="odometry")
    plot(p_state, p_action, p_obs, layout=(3,1), size=(600,600))
end

function show_map_estimation(map_est)
    println("Estimating uncertainty using information matrix...")
    symbols = [:drag, :mass, :wall_pos, Symbol("s0[1]"), Symbol("s0[2]")]
    cov_mat = @time informationmatrix(map_est)
    rows = map(symbols) do s
        v = map_est.values[s]
        std = cov_mat[s, s]
        (name=s, mean=v, std=std)
    end
    display(DataFrame(rows))
end

##
Δt = 0.1
tend = 10.0
times = 0:Δt:tend

##
# generate the data from the prior and plot

true_params = (s0=[0., 0.], drag=0.2, mass=2.0, wall_pos=11.0)
s0_dist = MvNormal([0., 0.], [0.01, 0.01])
process_prior = data_process(times; s0_dist, dyn_disturbance=false, noise_scale=0.01, true_params...)
prior_run = process_prior() 
process_infer = data_process(times; s0_dist, dyn_disturbance=false, noise_scale=0.01, prior_run.data...)
plot_result(prior_run, "data")

##

# MAP estimation using Turing
Turing.setadbackend(:forwarddiff)
println("Performing MAP estimation using Turing...")
map_est = @time let 
    trials = 10
    options = Optim.Options(x_abstol=1e-3, allow_f_increases=true)
    sols = [optimize(process_infer, MAP(), options) for _ in 1:trials]
    scores = [s.lp for s in sols]
    _, i = findmax(scores)
    @show scores
    sols[i]
end
map_values = map_est.values
dis_to_wall = map_values[:wall_pos] - map_values[Symbol("s0[1]")]
display(map_est)
@show dis_to_wall

show_map_estimation(map_est)

##
# Sampling full posterior
println("Sampling from the full posterior...")
chains = @time Turing.sample(
    process_infer, NUTS(), MCMCThreads(), 500, 8, ϵ=0.01, init_theta=map_est)
summarize(chains)

##
# Other sampling methods
chains = @time Turing.sample(
    process_infer, HMC(0.001, 5), MCMCThreads(), 20_000, 8, init_theta=map_est)
##

#------------ Outdated code below -------------
# MAP estimation using handcrafted score and plot the infered trajecotry
x_transform(n_times) = as((
    drag=as(Real, 0., 1.), 
    mass=as(Real, 0.5, 5.0), 
    wall_pos=as(Real, 0., 50.),
    states=as(Array, 2, n_times-1),
))

function MAP_infer(times, data; ad = AutoForwardDiff())
    x_trans = x_transform(length(times))
    function f(x, _)
        -posterior_density(transform(x_trans, x), times, data)
    end
    of = OptimizationFunction(f, ad)
    states0 = zeros(2, length(times)-1)
    x0 = (drag=0.1, mass=1.0, wall_pos=8.0, states=states0)
    v0 = inverse(x_trans, x0)
    prob = OptimizationProblem(of, v0)
    # sol = solve(prob, ADAM(0.1, (0.9, 0.9)), x_abstol=1e-3, progress=true, maxiters=4000)
    sol = solve(prob, LBFGS(), allow_f_increases=true)
    # sol = solve(OptimizationProblem(of, sol0.u), LBFGS(), x_abstol=1e-3, progress=true)
    # @show sol.minimum
    
    score = sol.minimum
    transform(x_trans, sol.u), score
end

println("Performing MAP estimation using log density...")
map_result, score = @time MAP_infer(times, prior_run.data, ad=AutoReverseDiff())
result_infered = (; map_result.states, times, map_result.wall_pos, prior_run.data)
let
    (; wall_pos, drag, mass) = map_result
    @show wall_pos
    @show drag
    @show mass
end

plot_result(result_infered, "infered")

##
# Full posterior sampling
using AdvancedHMC, ReverseDiff
using AdvancedHMC: MassMatrixAdaptor, StepSizeAdaptor
using DiffResults: DiffResult
function Post_sample(times, data, x0; euler)
    x_trans = x_transform(length(times))
    f(θ) = -posterior_density(θ, times, data; euler)
    fx(x) = f(transform(x_trans, x))
    dfdθ(θ) = begin
        res = DiffResults.GradientResult(θ)
        ReverseDiff.gradient!(res, f, θ)
        return DiffResults.value(res), DiffResults.gradient(res)
    end
    n_samples = 200
    n_adapts = n_samples ÷ 2
    metric = DiagEuclideanMetric(dimension(x_trans))
    hamiltonian = Hamiltonian(metric, f, dfdθ)

    θ0 = inverse(x_trans, x0)
    initial_ϵ = find_good_stepsize(hamiltonian, θ0)
    integrator = Leapfrog(initial_ϵ)

    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    sample(hamiltonian, proposal, θ0, n_samples, adaptor, n_adapts; progress=true)
end
samples, stats = @time Post_sample(times, prior_run.data, map_result, euler=false)


##

prior_run.data

map_params = [k => map_result[k] for k in [:drag, :mass, :wall_pos]]
function check_lp(step::Int)
    process_map = data_process(times[1:step]; actions=prior_run.data.actions, map_result...)
    chain_map = sample(process_map, Prior(), 1)
    chain_map[:lp]
end
process_run_map = 
    data_process(times[1:step]; actions=prior_run.data.actions, map_params...)()
plot_result(process_run_map)

##
