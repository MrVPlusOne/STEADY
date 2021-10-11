##-----------------------------------------------------------
using Distributions
using StatsPlots
using ProgressMeter
using StatsFuns: softmax, logsumexp, logsubexp
import Random
##-----------------------------------------------------------
function simulate_trajectory(times, (; x0_dist, motion_model, obs_model), controller)
    T = length(times)
    state = rand(x0_dist)

    states = []
    observations = []
    controls = []

    for t in 1:T
        y = rand(obs_model(state))
        u = controller(state, y)

        push!(states, state)
        push!(observations, y)
        push!(controls, u)

        if t < T
            Δt = times[t+1] - times[t]
            state = rand(motion_model(state, u, Δt))
        end
    end

    (states=specific_elems(states), 
        observations=specific_elems(observations), 
        controls=specific_elems(controls))
end

function states_likelihood(system, (;times, controls), states)
    (; x0_dist, motion_model) = system
    p = logpdf(x0_dist, states[1])
    for i in 1:length(states)-1
        Δt = times[i+1]-times[i]
        state_distr = motion_model(states[i], controls[i], Δt)
        p += logpdf(state_distr, states[i+1])
    end
    p    
end

function data_likelihood(system, (; observations), states)
    sum(logpdf(system.obs_model(states[i]), observations[i])
        for i in 1:length(states))
end

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

const wall_pos = 10.0
function car_obs_model(state)
    (pos, pos′) = state
    dis = wall_pos - pos
    v_mea = pos′
    SMvNormal(@SVector[dis, v_mea], @SVector[1.0, 0.3])
end

function car1d_system(params, x0_dist, f_x′′, σ::NamedTuple)
    StochasticSystem(x0_dist, car_motion_model(params; f_x′′, σ), car_obs_model; 
        X=typeof(rand(x0_dist)), Y=SVector{2, Float64}, U=SVector{1, Float64})
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
    @SVector[Car1D.controller((speed=z[2], sensor=z[1])).f]
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
columns(m::Matrix) = [m[:, i] for i in 1:size(m)[2]]
rows(m::Matrix) = [m[i, :] for i in 1:size(m)[1]]

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
map_params = fit_dynamics_params(
    Car1D.acceleration_f,
    subsample(ffbs_result.particles, 200), subsample(ffbs_result.log_weights, 200), 
    x0_dist, params_dist,
    obs_data, 
    (; mass=1.0, drag = 0.5),
    true_σ;
    optim_options = Optim.Options(f_abstol=1e-4),
)
##-----------------------------------------------------------
# iterative synthesis utilities
function sample_posterior_data(
    system::StochasticSystem, obs_data; 
    use_ffbs=true, n_particles, max_trajs, resample_threshold = 0.5,
    progress_offset=0,
)
    local (; particles, log_weights, log_obs) = 
        if use_ffbs
            ffbs_smoother(system, obs_data; 
                n_particles, n_trajs=max_trajs, resample_threshold, progress_offset)
        else
            filter_smoother(system, obs_data, n_particles; 
                thining=n_particles ÷ max_trajs, resample_threshold)
        end
    local trajectories = rows(particles)
    (; particles, log_weights, trajectories, log_obs)
end

function check_performance_variance(
    system, obs_data; 
    n_particles, n_trajs, use_ffbs, repeats=50, 
)
    @unzip_named (perfs, :perf), (stats, :stats) = map(1:repeats) do _
        print(".")
        sample_posterior_data(system, obs_data; n_particles, max_trajs=n_trajs, use_ffbs)
    end
    println()
    (; perf = to_measurement(perfs), stats=to_measurement(stats))
end

##-----------------------------------------------------------
# check posterior sampling variance
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
components_transcendentals!(comp_env, can_grow=true)

prior_p = let
    x₀, x′₀ = ex_data.states[1]
    syn_params = (;true_params.mass, true_params.drag)
    others = (; wall=wall_pos)
    logpdf(to_distribution(vdata), (;x₀, x′₀, params=syn_params, others))
end
if !isfinite(prior_p)
    error("prior_p = $prior_p, some value may be out of its support.")
end

sketch = no_sketch(vdata.state′′_vars)
rand(params_dist)
rand_inputs = [merge((pos = randn(), pos′ = randn(), f = 2randn()), rand(params_dist)) for _ in 1:200]
pruner = IOPruner(; inputs=rand_inputs, comp_env)
# pruner = RebootPruner(;comp_env.rules)
senum = @time synthesis_enumeration(
    vdata, sketch, Car1D.action_vars(), comp_env, 7; pruner)
let rows = map(senum.enum_result.pruned) do r
        (; r.pruned, r.reason)
    end
    show(DataFrame(rows), truncate=100)
    println()
end
display(senum)
##-----------------------------------------------------------
# test synthesis iteration
post_data=sample_posterior_data(true_system, obs_data; 
    n_particles=10_000, max_trajs=100)
optim_options = Optim.Options(
    f_abstol=1e-4,
    iterations=100,
    outer_iterations=40, 
    time_limit=2.0)
fit_settings = DynamicsFittingSettings(; optim_options)

@info "fit_dynamics started..."
syn_result = fit_dynamics(
    senum, post_data.particles, post_data.log_weights, obs_data, nothing; 
    σ_guess = true_σ,
    program_logp=prog_size_prior(0.2), fit_settings,
)
display(syn_result)
map(syn_result.sorted_results) do (; score, f_x′′, params)
    (score, f_x′′.ast, params)
end |> DataFrame
##-----------------------------------------------------------
# test full synthesis
@info "full synthesis started..."
# mini_sketch = DynamicsSketch(
#     [Var(:drag_acc, ℝ, PUnits.Acceleration)], 
#     ((; mass, f), (; drag_acc)) -> (pos′′= f / mass - drag_acc,),
# )

iter_result = let senum = synthesis_enumeration(
        vdata, sketch, Car1D.action_vars(), comp_env, 7; pruner)
    
    comps_guess = let f = Var(:f, ℝ, PUnits.Force), mass = Var(:mass, ℝ, PUnits.Mass) 
        (f / mass, )
    end
    params_guess = nothing # (mass=3.0, drag=0.8,)
    σ_guess = big_σ

    display(senum.enum_result[sketch.holes[1].type] |> collect)

    @time fit_dynamics_iterative(senum, obs_data, comps_guess, params_guess, σ_guess;
        program_logp=prog_size_prior(0.2), fit_settings,
        max_iters=100,
    )
end
plot(iter_result.logp_history, xlabel="iterations", title="log p(observations)", 
    legend=false) |> display
let rows = []
    for (i, (f, params)) in enumerate(iter_result.dyn_history)
        (i % 5 == 1) && push!(rows, (; i, f.ast.pos′′, params))
    end
    show(DataFrame(rows), truncate=100)
end
##-----------------------------------------------------------