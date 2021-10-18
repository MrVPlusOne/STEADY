##-----------------------------------------------------------
using Distributions
using StatsPlots
using ProgressMeter
using StatsFuns: softmax, logsumexp, logsubexp
import Random
##-----------------------------------------------------------
plot_states!(plt, states, times, name) = begin
    @unzip xs, vs = states
    plot!(plt, times, hcat(xs, vs), label=["x ($name)" "v ($name)"])
end
plot_states(states, times, name) = 
    plot_states!(plot(legend=:outerbottom), states, times, name)

plot_particles(particles::Matrix, times, name, true_states=nothing) = 
    let particles = (to_svec ∘ values).(particles)
        @unzip y_mean, y_std = map(1:length(times)) do t
            ys = particles[:, t]
            mean(ys), 2std(ys)
        end
        to_data(vecs) = reduce(hcat, vecs)'
        p = plot(times, to_data(y_mean), ribbon=to_data.((y_std, y_std)), 
            label=["x ($name)" "v ($name)"])
        if true_states !== nothing
            plot_states!(p, true_states, times, "truth")
        end
        plot(p, legend=:outerbottom)
    end

subsample(::Nothing, n_max) = nothing
function subsample(xs, n_max)
    N = min(n_max, size(xs,1))
    thining = size(xs,1) ÷ N
    rank = length(size(xs)) 
    if rank == 1
        xs[1:thining:end]
    elseif rank == 2
        xs[1:thining:end, :]
    else
        error("not implemented for xs=$xs")
    end
end

function plot_trajectories(trajectories, times, name; n_max=50)
    trajectories = subsample(trajectories, n_max)
    @unzip xs, vs = trajectories
    p1 = plot(times, xs')
    p2 = plot(times, vs')
    plot(p1, p2, layout=(2,1), size=(600, 800), legend=false)
end

function car_motion_model((; drag, mass); 
    f_x′′=Car1D.acceleration_f, σ=(σ_pos=0.1, σ_pos′=0.2),
)
    (state, ctrl, Δt) -> begin
        (pos, pos′) = state
        f = ctrl[1]
        (; pos′′) = f_x′′((;pos, pos′, f, drag, mass))
        DistrIterator(
            (pos=SNormal(pos + Δt * pos′, σ.σ_pos * Δt),
            pos′=SNormal(pos′ + Δt * pos′′, (abs(pos′) + 0.1) * σ.σ_pos′ * Δt)))
    end
end

#TODO: refactor to use the new sketch API.
car1d_sketch((; σ_pos, σ_pos′)) = DynamicsSketch(
    [Var(:pos′′, ℝ, PUnits.Acceleration)],
    (inputs, (; pos′′)) -> begin
        (; Δt, pos, pos′) = inputs
        DistrIterator(
            (pos=SNormal(pos + Δt * pos′, σ_pos * Δt),
            pos′=SNormal(pos′ + Δt * pos′′, (abs(pos′) + 0.1) * σ_pos′ * Δt)))
    end
)

const wall_pos = 10.0
function car_obs_model(state)
    (pos, pos′) = state
    dis = wall_pos - pos
    v_mea = pos′
    DistrIterator((dis=Normal(dis, 1.0), v_mea=Normal(v_mea, 0.3)))
end

function car1d_system(params, x0_dist, f_x′′, σ::NamedTuple)
    MarkovSystem(x0_dist, car_motion_model(params; f_x′′, σ), car_obs_model)
end
##-----------------------------------------------------------
# generate data
Random.seed!(123)
times = collect(0.0:0.1:10.0)
T = length(times)
true_params = (; mass=1.5, drag = 0.5) #, σ_pos=0.1, σ_pos′=0.2)
true_σ = (σ_pos=0.1, σ_pos′=0.2)
bad_params = (; mass=6.0, drag = 0.85)
big_σ = (σ_pos=0.2, σ_pos′=0.4)
car_controller = (s, z) -> begin
    Car1D.controller((speed=z[2], sensor=z[1]))
end
vdata = Car1D.variable_data()
x0_dist = init_state_distribution(vdata)
params_dist = params_distribution(vdata)
true_system = car1d_system(true_params, x0_dist, Car1D.acceleration_f, true_σ)
bad_system = car1d_system(bad_params, x0_dist, Car1D.acceleration_f, true_σ)
true_motion_model = true_system.motion_model

ex_data = simulate_trajectory(times, true_system, car_controller)
obs_data = (; times, ex_data.observations, ex_data.controls)

plot_states(ex_data.states, times, "truth") |> display
##-----------------------------------------------------------
# particle filter inference
function particle_trajectories(particles::Matrix{P}, ancestors::Matrix{Int}) where P
    N, T = size(particles)
    trajs = Array{P}(undef, size(particles))
    trajs[:, T] = particles[:, T]
    indices = collect(1:N)
    for t in T-1:-1:1
        indices .= ancestors[indices, t+1]
        trajs[:, t] = particles[indices, t]
    end
    trajs
end


function check_log_scores(ls, name)
    ess = effective_particles(softmax(ls))
    histogram(ls, ylabel="log scores", title=name) |> display
    ratio = exp(maximum(ls) - minimum(ls))
    (; ess, ratio, name)
end
##-----------------------------------------------------------
# test filters and smoothers
pf_result = @time forward_filter(bad_system, obs_data, 10000, resample_threshold=1.0)
plot_particles(pf_result.particles, times, "particle filter", ex_data.states) |> display

pfs_result = @time filter_smoother(bad_system, obs_data, 10000; 
    resample_threshold=0.5)
plot_particles(pfs_result[1], times, "filter-smoother", ex_data.states) |> display

ffbs_result = @time ffbs_smoother(bad_system, obs_data; n_particles=10_000, n_trajs=200,
    resample_threshold=0.5)
plot_particles(ffbs_result[1], times, "FFBS", ex_data.states) |> display
##-----------------------------------------------------------
# simple test
map_params = @time fit_dynamics_params(
    car_motion_model,
    subsample(ffbs_result.particles, 200), subsample(ffbs_result.log_weights, 200), 
    x0_dist, params_dist,
    obs_data, 
    (; mass=1.0, drag = 0.5);
    optim_options = Optim.Options(f_abstol=1e-4),
)
##-----------------------------------------------------------
# check posterior sampling variance
function check_performance_variance(
    system, obs_data; 
    n_particles, n_trajs, use_ffbs, repeats=50, 
)
    @unzip_named (perfs, :perf), (stats, :stats) = map(1:repeats) do _
        print(".")
        ffbs_smoother(system, obs_data; n_particles, n_trajs)
    end
    println()
    (; perf = to_measurement(perfs), stats=to_measurement(stats))
end

skip_variance_checking = true
if !skip_variance_checking
    check_performance_variance(bad_system, obs_data;
        n_particles=10_000, n_trajs=200, use_ffbs=false)
    check_performance_variance(bad_system, obs_data;
        n_particles=10_000, n_trajs=200, use_ffbs=true)
end
##-----------------------------------------------------------
# program enumeration test
shape_env = ℝenv()
comp_env = ComponentEnv()
components_scalar_arithmatic!(comp_env, can_grow=true)
components_special_functions!(comp_env, can_grow=true)

good_sketch = car1d_sketch(big_σ)

prior_p = let
    x₀ = ex_data.states[1]
    syn_params = (;true_params.mass, true_params.drag)
    logpdf(to_distribution(vdata), (;x₀, params=syn_params))
end
if !isfinite(prior_p)
    error("prior_p = $prior_p, some value may be out of its support.")
end

rand_inputs = [merge((pos = randn(), pos′ = randn(), f = 2randn()), rand(params_dist)) for _ in 1:200]
pruner = IOPruner(; inputs=rand_inputs, comp_env)
# pruner = RebootPruner(;comp_env.rules)
senum = @time synthesis_enumeration(vdata, good_sketch, shape_env, comp_env, 7; pruner)
let rows = map(senum.enum_result.pruned) do r
        (; r.pruned, r.reason)
    end
    show(DataFrame(rows), truncate=100)
    println()
end
display(senum)
##-----------------------------------------------------------
# test synthesis iteration
post_data=ffbs_smoother(true_system, obs_data; n_particles=10_000, n_trajs=100)
optim_options = Optim.Options(
    f_abstol=1e-4,
    iterations=100,
    outer_iterations=40, 
    time_limit=2.0)
fit_settings = DynamicsFittingSettings(; optim_options)

@info "fit_dynamics started..."
syn_result = fit_dynamics(
    senum, post_data.particles, post_data.log_weights, obs_data, nothing; 
    program_logp=prog_size_prior(0.2), fit_settings,
)
display(syn_result)
map(syn_result.sorted_results) do (; score, comps, params)
    (score, comps, params)
end |> DataFrame
##-----------------------------------------------------------
# optionally config the full synthesis
@info "full synthesis started..."
infer_variance = true
if infer_variance
    good_sketch = DynamicsSketch(
        [Var(:pos′′, ℝ, PUnits.Acceleration)],
        (inputs, (; pos′′)) -> begin
            (; Δt, pos, pos′, σ_pos, σ_pos′) = inputs
            DistrIterator(
                (pos=SNormal(pos + Δt * pos′, σ_pos * Δt),
                pos′=SNormal(pos′ + Δt * pos′′, (abs(pos′) + 0.1) * σ_pos′ * Δt)))
    end)

    vdata.dynamics_params[Var(:σ_pos, ℝ, PUnits.Length)] = PertBeta(0.0, 0.25, 0.5)
    vdata.dynamics_params[Var(:σ_pos′, ℝ, PUnits.Speed)] = PertBeta(0.0, 0.25, 0.5)
end
##-----------------------------------------------------------
# run full synthesis
iter_result = let senum = synthesis_enumeration(
        vdata, good_sketch, shape_env, comp_env, 7; pruner)
    
    comps_guess = let f = Var(:f, ℝ, PUnits.Force), mass = Var(:mass, ℝ, PUnits.Mass) 
        (pos′′ = f / mass,)
    end
    params_guess = nothing # (mass=3.0, drag=0.8,)

    display(senum.enum_result[good_sketch.holes[1].type] |> collect)

    @time fit_dynamics_iterative(senum, obs_data, comps_guess, params_guess;
        obs_model=car_obs_model,
        program_logp=prog_size_prior(0.2), fit_settings,
        max_iters=600,
    )
end
plot(iter_result.logp_history, xlabel="iterations", title="log p(observations)", 
    legend=false) |> display

let 
    (; logp_history, improve_pred_hisotry) = iter_result
    improve_actual = logp_history[2:end] .- logp_history[1:end-1]
    data = hcat(improve_actual, improve_pred_hisotry) |> specific_elems
    @show data
    plot(data, xlabel="iterations", ylabel="iteration improvement", 
        label=["est. actual" "predicted"], ylims=[-0.5, 0.5])
    hline!([0.0], label="y=0", line=:dash) |> display
end
let rows=[], step=length(iter_result.dyn_history)÷10
    for (i, (; comps, params)) in enumerate(iter_result.dyn_history)
        (i % 50 == 1) && push!(rows, (; i, comps, params))
    end
    show(DataFrame(rows), truncate=100)
end
##-----------------------------------------------------------