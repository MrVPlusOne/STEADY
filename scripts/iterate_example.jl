##-----------------------------------------------------------
using Distributions
using StatsPlots
using ProgressMeter
using StatsFuns: softmax, logsumexp, logsubexp
# using Gen
# using LowLevelParticleFilters
# import LowLevelParticleFilters as PF
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

function states_likelihood(states, times, x0_dist, motion_model, controls)
    p = logpdf(x0_dist, states[1])
    for i in 1:length(states)-1
        Δt = times[i+1]-times[i]
        state_distr = motion_model(states[i], controls[i], Δt)
        p += logpdf(state_distr, states[i+1])
    end
    p    
end

function data_likelihood(states, obs_model, observations)
    sum(logpdf(obs_model(states[i]), observations[i])
        for i in 1:length(states))
end

plot_states!(plt, states, times, name) = begin
    plot!(plt, times, reduce(hcat, states)', label=["x ($name)" "v ($name)"])
end
plot_states(states, times, name) = 
    plot_states!(plot(legend=:outerbottom), states, times, name)

plot_particles(particles::Matrix, times, name, true_states=nothing) = let
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

function car_motion_model((; drag, mass, σ_pos, σ_pos′))
    (state, ctrl, Δt) -> begin
        (pos, pos′) = state
        f = ctrl[1]
        (; pos′′) = Car1D.acceleration_f((;f, drag, mass, pos′))
        μ = @SVector[pos + Δt * pos′,  pos′ + Δt * pos′′]
        SMvNormal(μ, @SVector[σ_pos * Δt, (abs(pos′) + 0.1) * σ_pos′ * Δt])
    end
end

const wall_pos = 10.0
function car_obs_model(state)
    (pos, pos′) = state
    dis = wall_pos - pos
    v_mea = pos′
    SMvNormal(@SVector[dis, v_mea], @SVector[1.0, 0.3])
end

function params_to_system(params)
    StochasticSystem(x0_dist, car_motion_model(params), car_obs_model; 
        X=SVector{2, Float64}, Y=SVector{2, Float64}, U=SVector{1, Float64})
end
##-----------------------------------------------------------
# generate data
Random.seed!(123)
times = collect(0.0:0.1:8.0)
T = length(times)
params = (; drag = 0.2, mass=1.5, σ_pos=0.1, σ_pos′=0.2)
car_controller = (s, z) -> begin
    @SVector[Car1D.controller((speed=z[2], sensor=z[1])).f]
end
true_motion_model = car_motion_model(params)
x0_dist = SMvNormal(@SVector[0.0, 0.0], @SVector[0.01, 0.5])
true_system = params_to_system(params)

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

function filter_smoother(system, data, n_particles; 
        resample_threshold=1.0, thining=10, track_weights=true)
    pf = track_weights ? forward_smoother : forward_filter
    (; particles, log_weights, ancestors) = pf(system, data, n_particles; resample_threshold)
    trajs = particle_trajectories(particles, ancestors)
    trajectories = trajs[1:thining:end,:]
    log_weights = log_weights[1:thining:end, end]
    log_weights .-= logsumexp(log_weights)
    (; trajectories, log_weights)
end

function check_log_scores(ls, name)
    ess = effective_particles(softmax(ls))
    histogram(ls, ylabel="log scores", title=name) |> display
    ratio = exp(maximum(ls) - minimum(ls))
    (; ess, ratio, name)
end
##-----------------------------------------------------------
# test filters and smoothers
pf_result = @time forward_filter(true_system, obs_data, 10000)
plot_particles(pf_result.particles, times, "particle filter", ex_data.states) |> display

pfs_result = @time filter_smoother(true_system, obs_data, 10000; 
    resample_threshold=1.0, track_weights=true)
plot_particles(pfs_result[1], times, "filter-smoother", ex_data.states) |> display

plot_trajectories(pfs_result[1], times, "filter-smoother", n_max=10) |> display

check_log_scores(pfs_result[2], "trajectory weights")
let dl = data_likelihood.(rows(pfs_result[1]), car_obs_model, Ref(obs_data.observations))
    check_log_scores(pfs_result[2]-dl, "neg data likelihood")
end
let sl = map(rows(subsample(pfs_result[1], 50))) do r
        states_likelihood(r, times, x0_dist, true_motion_model, obs_data.controls)
    end
    check_log_scores(sl, "state likelihood")
end
let dl = data_likelihood.(rows(pfs_result[1]), car_obs_model, Ref(obs_data.observations)), \
    sl = map(rows(pfs_result[1])) do r
        states_likelihood(r, times, x0_dist, true_motion_model, obs_data.controls)
    end
    check_log_scores(dl .+ sl, "total likelihood")
end

let lws = [sum(x -> logpdf(Normal(), x), rand(Normal()) for _ in 1:100) for _ in 1:100]
    check_log_scores(lws, "random walk")
end
##-----------------------------------------------------------
# dynamics fitting
to_params(vec) = let
    local drag, mass = vec#, σ_pos, σ_pos′ = vec
    (; drag, mass, σ_pos=0.1, σ_pos′=0.2) # σ_pos= abs(σ_pos), σ_pos′=abs(σ_pos′))
end

to_vec((; drag, mass, σ_pos, σ_pos′)) = [drag, mass]#, σ_pos, σ_pos′]

# to_params(vec) = let
#     local drag, mass = vec
#     (; drag, mass, σ_pos=0.1, σ_pos′=0.2)
# end

# to_vec((; drag, mass, σ_pos, σ_pos′)) = [drag, mass]

function expected_data_p(log_prior, log_data_p, log_state_p; debug = false)
    # compute the importance wights of the samples under the new dynamics        
    local log_weights = log_softmax(log_prior .- log_data_p)
    if debug
        n_samples = length(log_prior)
        max_ratio = exp(maximum(log_weights) - minimum(log_weights))
        ess_prior = effective_particles(softmax(log_prior))
        ess_data_inv = effective_particles(softmax(-log_data_p))
        ess = effective_particles(softmax(log_weights .+ log_data_p))
        @info expected_data_p n_samples ess ess_data_inv ess_prior max_ratio
    end
    # compute the (weighted) expectated data log probability
    logsumexp(log_weights .+ log_data_p)
end

"""
Compute the posterior average log data likelihood, ``∫ log(p(y|x)) p(x|y,f) dx``.
"""
function expected_log_p(log_prior, log_data_p, log_state_p; debug = false)
    # compute the importance wights of the samples under the new dynamics        
    local weights = softmax(log_prior)
    if debug
        max_ratio = maximum(weights)/minimum(weights)
        ess = effective_particles(weights)
        @info expected_log_p ess max_ratio
    end
    # compute the (weighted) expectated data log probability
    sum(weights .* (log_data_p + log_state_p))
end

function sample_performance(log_prior, log_data_p, log_state_p; debug = false)
    expected_log_p(log_prior, log_data_p, log_state_p; debug = false)
end

function fit_dynamics(
        particles::Matrix, log_weights::Vector, times, x0_dist, (; controls, observations), 
        params_guess; λ::Float64)
    trajectories = rows(particles)
    data_scores = data_likelihood.(
        trajectories, Ref(car_obs_model), Ref(observations))
    state_scores = states_likelihood.(
        trajectories, Ref(times), Ref(x0_dist), Ref(car_motion_model(params_guess)), Ref(controls))

    function objective(vec; use_λ::Bool=true)
        local params = to_params(vec)
        local prior = logpdf(Normal(0.0, 0.5), params.σ_pos) + logpdf(Normal(0.0, 0.5), params.σ_pos′)
        local state_scores′ = states_likelihood.(
            trajectories, Ref(times), Ref(x0_dist), Ref(car_motion_model(params)), Ref(controls))

        local perf = sample_performance(#=state_scores′ - state_scores=# + log_weights, 
            data_scores, state_scores′)
        local kl = 0.5mean((state_scores′ .- state_scores).^2)
        prior + perf - use_λ * λ * kl
    end

    f_init = objective(params_guess; use_λ=false)
    alg = Optim.LBFGS()
    sol = Optim.maximize(objective, to_vec(params_guess), alg, autodiff=:forward)
    Optim.converged(sol) || display(sol)
    f_final = objective(Optim.maximizer(sol); use_λ=false)
    params = Optim.maximizer(sol) |> to_params
    (; f_init, f_final, params)
end

as_real(x::Real) = x
as_real(x::Dual) = x.value
##-----------------------------------------------------------
# simple test
map_params = fit_dynamics(subsample(pfs_result[1], 200), subsample(pfs_result[2], 200), 
    times, x0_dist, obs_data,
    (; drag = 0.5, mass=1.0, σ_pos=0.2, σ_pos′=0.4); 
    λ=1.0)
##-----------------------------------------------------------
# iterative synthesis utilities
function iterative_dyn_fitting(params₀, obs_data, iters; 
        n_particles=10000, max_trajs=500, trust_thresholds=(0.25, 0.75), plot_freq=10)
    
    function sample_data(params; debug=false)
        local system = params_to_system(params)
        local particles, log_weights = filter_smoother(system, obs_data, n_particles; 
            thining=1, track_weights=true, resample_threshold=1.0)
        local trajectories = rows(particles)
        local dl = data_likelihood.(trajectories, Ref(car_obs_model), Ref(obs_data.observations))
        local sl = map(trajectories) do tr 
            states_likelihood(
                tr, times, system.x0_dist, system.motion_model, obs_data.controls)
        end
        local perf = sample_performance(log_weights, dl, sl; debug)
        particles, log_weights, trajectories = subsample.(
            (particles, log_weights, trajectories), Ref(max_trajs))
        (; particles, log_weights, trajectories, perf)
    end
    
    params_est = params₀

    params_history = typeof(params_est)[]
    traj_history = []
    score_history = []
    λ_history = []
    λ = 1.0
    @showprogress for i in 1:iters
        (; particles, log_weights, perf, trajectories) = sample_data(params_est, debug=true)
        perf2 = sample_data(params_est).perf

        fit_r = fit_dynamics(particles, log_weights, times, x0_dist, obs_data, params_est; λ)
        perf_new = sample_data(fit_r.params).perf

        begin
            improve_actual = perf_new - perf
            improve_pred = fit_r.f_final - fit_r.f_init
            ρ = improve_actual / improve_pred
            @info "optimization finished." ρ improve_actual improve_pred
            
            perfΔ = abs(perf2 - perf)
            if perfΔ > 0.5 * improve_pred
                @info "High performance variance" perfΔ
            end
            λ_ratio = 1.5
            if ρ > trust_thresholds[2]
                λ /= λ_ratio
            elseif ρ < trust_thresholds[1]
                λ *= λ_ratio
            end
        end
        if ρ < 0
            @warn("Failed to improve the objective, reject the move.")
        else
            params_est = fit_r.params
        end
        push!(params_history, params_est)
        push!(traj_history, trajectories)
        push!(score_history, perf)
        push!(λ_history, λ)
        @show "iteration finished" λ perf params_est
        if i % plot_freq == 1
            plot_particles(particles, times, "iteration $i", ex_data.states) |> display
        end
    end
    (; params_history, traj_history, score_history)
end
##-----------------------------------------------------------
# perform iterative synthesis
bad_params = (; drag = 1.0, mass=6.0, σ_pos=0.4, σ_pos′=0.8)
# bad_params = (; drag = 0.254553, mass= 1.42392, σ_pos=0.31645, σ_pos′=0.584295)
syn_result = iterative_dyn_fitting(params, obs_data, 200, n_particles=10000, max_trajs=500)
show(DataFrame(syn_result.params_history), allrows=true)
plot(syn_result.score_history, title="log p(y|f)")
##-----------------------------------------------------------
# full program synthesis
shape_env = ℝenv()
comp_env = ComponentEnv()
components_scalar_arithmatic!(comp_env, can_grow=false)

vdata = Car1D.variable_data()

prior_p = let
    x₀, x′₀ = ex_data.states[1]
    syn_params = (;params.mass, params.drag)
    others = (; wall=wall_pos)
    logpdf(to_distribution(vdata), (;x₀, x′₀, params=syn_params, others))
end
if !isfinite(prior_p)
    error("prior_p = $prior_p, some value may be out of its support.")
end

sketch = no_sketch(vdata.state′′_vars)
prog_logp(comps) = log(0.5) * sum(ast_size.(comps))  # weakly penealize larger programs

params_dist = params_distribution(vdata)
rand(params_dist)
rand_inputs = [merge((pos = randn(), pos′ = randn(), f = 2randn()), rand(params_dist)) for _ in 1:100]
pruner = IOPruner(; inputs=rand_inputs, comp_env)
# pruner = RebootPruner(;comp_env.rules)
senum = @time synthesis_enumeration(
    vdata, sketch, Car1D.action_vars(), comp_env, 9; pruner)
let rows = map(senum.enum_result.pruned) do r
        (; r.pruned, r.reason)
    end
    show(DataFrame(rows), truncate=100)
    println()
end
display(senum)
##-----------------------------------------------------------