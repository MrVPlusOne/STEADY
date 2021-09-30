##-----------------------------------------------------------
using Gen
using StatsPlots
using ProgressMeter
using StatsFuns: softmax, logsumexp, logsubexp
using LowLevelParticleFilters
import LowLevelParticleFilters as PF
##-----------------------------------------------------------
@gen function data_process(gen_mode,
        times, motion_model, obs_model, controller, controls)
    local states
    T = length(times)-1
    Num = Float64
    state = @trace broadcasted_normal([0.0, 0.0], [0.01, 0.5]) (:state, 1)
    if gen_mode
        states = Matrix{Num}(undef, 2, T)
        observations = Matrix{Num}(undef, 2, T)
        controls = Matrix{Num}(undef, 1, T)
    end

    for i in 1:T
        Δt = times[i+1]-times[i]
        obs_distr = obs_model(state)
        obs = @trace broadcasted_normal(obs_distr[1], obs_distr[2]) (:obs, i)
        if gen_mode
            states[:, i] = state
            observations[:, i] = obs
            controls[:, i] = controller(state, obs)
        end
        state_distr = motion_model(state, controls[i], Δt)
        if i < T
            state = @trace broadcasted_normal(state_distr[1], state_distr[2]) (:state, i+1)
        end
    end
    gen_mode && return (;states, observations, controls)
end

function states_likelihood(trace)
    (gen_mode, times, motion_model, obs_model, controller, controls) = get_args(trace)
    states = get_states(trace, length(times))
    states_likelihood(times, motion_model, states, controls)
end

function states_likelihood(times, motion_model, states, controls)
    s = logpdf(MvNormal([0.0, 0.0], [0.01, 0.5]), states[1])
    for i in 1:length(states)-1
        Δt = times[i+1]-times[i]
        state_distr = MvNormal(motion_model(states[i], controls[i], Δt)...)
        s += logpdf(state_distr, states[i+1])
    end
    s    
end

function data_likelihood(trace)
    gen_mode, times = get_args(trace)
    T = length(times)
    states = get_states(trace, T)
    data_likelihood(states, obs_model, get_observations(trace, T))
end

function data_likelihood(states, obs_model, observations)
    sum(logpdf(MvNormal(obs_model(states[i])...), observations[i])
        for i in 1:length(states))
end

function get_states(trace, T)
    cmap = get_choices(trace)
    [cmap[(:state, i)] for i in 1:T-1]
end

function get_observations(trace, T)
    cmap = get_choices(trace)
    [cmap[(:obs, i)] for i in 1:T-1]
end

plot_trace(trace, times, name) = plot_trace!(plot(legend=:outerbottom), trace, times, name)
plot_trace!(p, trace, times, name) = let
    times = times[1:end-1]
    cmap = get_choices(trace)
    states = (cmap[(:state, i)] for i in 1:length(times))
    plot!(p, times, reduce(hcat, states)', label=["x ($name)" "v ($name)"])
end

plot_traces(traces, times, name, truth_trace=nothing) = let
    times = times[1:end-1]
    cmaps = get_choices.(traces)
    @unzip y_mean, y_std = map(1:length(times)) do i
        ys = (x -> x[(:state, i)]).(cmaps)
        mean(ys), std(ys)
    end
    to_data(vecs) = reduce(hcat, vecs)'
    p = plot(times, to_data(y_mean), ribbon=to_data.((y_std, y_std)), 
        label=["x ($name)" "v ($name)"])
    if truth_trace !== nothing
        plot_trace!(p, truth_trace, times, "truth")
    end
    plot(p, legend=:outerbottom)
end


plot_particles(particles::Matrix, times, name, truth_trace=nothing) = let
    times = times[1:end-1]
    @unzip y_mean, y_std = map(1:length(times)) do i
        ys = particles[:, i]
        mean(ys), std(ys)
    end
    to_data(vecs) = reduce(hcat, vecs)'
    p = plot(times, to_data(y_mean), ribbon=to_data.((y_std, y_std)), 
        label=["x ($name)" "v ($name)"])
    if truth_trace !== nothing
        plot_trace!(p, truth_trace, times, "truth")
    end
    plot(p, legend=:outerbottom)
end

function car_motion_model((; drag, mass, σ_pos, σ_pos′))
    (state, ctrl, Δt) -> begin
        (pos, pos′) = state
        f = ctrl[1]
        (; pos′′) = Car1D.acceleration_f((;f, drag, mass, pos′))
        μ = [pos + Δt * pos′,  pos′ + Δt * pos′′]
        (μ, abs.([σ_pos * Δt, σ_pos′ * Δt]))
    end
end

function obs_model(state)
    (pos, pos′) = state
    dis = 10.0 - pos
    v_mea = pos′
    ([dis, v_mea], [1.0, 0.3])
end
##-----------------------------------------------------------
# generate data
Random.seed!(123)
times = collect(0.0:0.1:8.0)
T = length(times)
params = (; drag = 0.2, mass=1.5, σ_pos=0.1, σ_pos′=0.2)
controller = (s, z) -> begin
    @SVector[Car1D.controller((speed=z[2], sensor=z[1])).f]
end
ground_motion_model = car_motion_model(params)
ex_trace, p_data = generate(data_process, 
    (true, times, 
    ground_motion_model,
    obs_model,
    controller,
    nothing),
)
ex_data = get_retval(ex_trace)
plot_trace(ex_trace, times, "truth") |> display
##----------------------------------------------------------- 
# randomly guess the states
obs_map = let
    content = (((:obs, i), ex_trace[(:obs, i)]) for i in 1:T-1)
    cmap = choicemap(content...)
end
post_args = (false, times, 
    ground_motion_model,
    obs_model,
    nothing,
    ex_data.controls)
guessed_trace, p_guess = generate(
    data_process,
    post_args,
    obs_map,
)
plot_trace(guessed_trace, times, "guess") |> display

(p_data, p_guess)
hidden_variables = Gen.select(((:state, i) for i in 1:T-1)...)
##-----------------------------------------------------------
# particle filter inference
function PF_inference_gen((;times, obs_map, controls), motion_model, n_particles)
    init_args = (false, times[1], motion_model, obs_model, nothing, controls)
    pf_state = initialize_particle_filter(data_process, init_args, choicemap(), n_particles)
    @showprogress for t in 2:length(times)
        maybe_resample!(pf_state)
        new_obs = choicemap(((:obs, i) => obs_map[(:obs, i)] for i in t-1:t-1)...)
        new_args = (false, times[1:t], motion_model, obs_model, nothing, controls)
        argdiffs = (NoChange(), UnknownChange(), NoChange(), NoChange(), NoChange(), NoChange())
        particle_filter_step!(pf_state, new_args, argdiffs, new_obs)
        (pf_state, new_args, argdiffs, new_obs)
    end
    pf_state
end

function make_particle_filter(N, motion_model, times)
    function dynamics(x,u,t,noise=false) 
        local Δt = times[t+1]-times[t]
        local x̂, σ = motion_model(x, u, Δt)
        if noise
            x̂ + rand(MvNormal(zero(x̂), σ))
        else
            x̂
        end
    end

    function dynamics_likelihood(x, u, x′, t)
        local Δt = times[t+1]-times[t]
        local x̂, σ = motion_model(x, u, Δt)
        logpdf(MvNormal(x̂, σ), x′)
    end
    
    function measurement(x,t,noise=false) 
        local ŷ, σ = obs_model(x)
        noise && (ŷ += rand(MvNormal(zero(ŷ), σ)))
        ŷ
    end

    function measurement_likelihood(x,u,y,t)
        local ŷ, σ = obs_model(x)
        logpdf(MvNormal(ŷ, σ), y)
    end

    d0 = MvNormal([0.0, 0.0], [0.01, 0.5])   # Initial state Distribution

    AdvancedParticleFilter(
        N, dynamics, measurement, measurement_likelihood, dynamics_likelihood, d0)
end

function PF_inference((;times, observations, controls), motion_model, n_particles)
    N = n_particles # Number of particles
    forward_trajectory(apf, controls, observations)
end

columns(m::Matrix) = [m[:, i] for i in 1:size(m)[2]]
rows(m::Matrix) = [m[i, :] for i in 1:size(m)[1]]

apf = make_particle_filter(50000, ground_motion_model, times)
pf_result = @time forward_trajectory(apf, columns(ex_data.controls), columns(ex_data.observations))
plot_particles(pf_result[1], times, "particle filter", ex_trace)

apf = make_particle_filter(1000, ground_motion_model, times)
ps_result = @time smooth(apf, 100, columns(ex_data.controls), columns(ex_data.observations))
plot_particles(ps_result[1], times, "particle smoother", ex_trace)

pf_s = PF_inference_gen((;times, obs_map, ex_data.controls), ground_motion_model, 500)
plot_traces(pf_s.traces[1:100], times, "particle filter", ex_trace)
##-----------------------------------------------------------
# dynamics fitting
to_params(vec) = let
    local drag, mass, σ_pos, σ_pos′ = vec
    (; drag, mass, σ_pos, σ_pos′)
end

to_vec((; drag, mass, σ_pos, σ_pos′)) = [drag, mass, σ_pos, σ_pos′]

function log_softmax(x::AbstractArray{<:Real})
    u = maximum(x)
    x = x .- u
    dnum = logsumexp(x)
    x .- dnum
end

let x = rand(5)
    exp.(log_softmax(x)) ≈ softmax(x) 
end

function sample_performance(log_joint_new, log_joint_old, log_data_p)
    # compute the importance wights of the samples under the new dynamics        
    local log_weights = log_softmax(log_joint_new .- log_joint_old .- log_data_p)
    # compute the (weighted) expectated data log probability
    logsumexp(log_weights .+ log_data_p)
end

function fit_dynamics(traces, times, controls, T, params_guess)
    trajectories = get_states.(traces, Ref(T))
    data_scores = data_likelihood.(traces)
    old_s = get_score.(traces)
    state_scores = states_likelihood.(traces)

    function objective(vec)
        local params = to_params(vec)
        local state_s = map(trajectories) do states
            states_likelihood(times, car_motion_model(params), states, controls)
        end
        local joint_s = state_s .+ data_scores

        local perf = sample_performance(joint_s, old_s, data_scores)
        local kl = 0.5mean((joint_s .- old_s).^2)
        perf - kl
    end

    f_init = objective(params_guess)
    alg = Optim.LBFGS()
    sol = Optim.maximize(objective, to_vec(params_guess), alg, autodiff=:forward)
    Optim.converged(sol) || display(sol)
    f_final = objective(Optim.maximizer(sol))
    params = Optim.maximizer(sol) |> to_params
    (; f_init, f_final, params)
end

function fit_dynamics(
        particles::Matrix, times, (; controls, observations), params_guess; λ=1.0)
    trajectories = rows(particles)
    data_scores = data_likelihood.(
        trajectories, Ref(obs_model), Ref(columns(observations)))
    state_scores = states_likelihood.(
        Ref(times), Ref(car_motion_model(params_guess)), trajectories, Ref(controls))

    function objective(vec; use_λ::Bool=true)
        local params = to_params(vec)
        local state_scores′ = states_likelihood.(
            Ref(times), Ref(car_motion_model(params)), trajectories, Ref(controls))

        local perf = sample_performance(state_scores′, state_scores, data_scores)
        local kl = 0.5mean((state_scores′ .- state_scores).^2)
        perf - use_λ * λ * kl
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
# show some rollouts
map_params = fit_dynamics(ps_result[1], times, ex_data,
    (; drag = 0.5, mass=1.0, σ_pos=0.2, σ_pos′=0.4))

let 
    rollouts = [generate(data_process, 
        (true, times, 
        car_motion_model(map_params),
        obs_model,
        controller,
        ex_data.controls),
    )[1] for _ in 1:20]
    plot_traces(rollouts, times, "map rollouts", ex_trace) |> display
end
##-----------------------------------------------------------
# iterative synthesis utilities
function select_elites(traces, elite_ratio)
    traces = traces |> sort_by(get_score) |> reverse
    local n = ceil(Int, length(traces)*elite_ratio)
    traces[1:n]
end

function iterative_dyn_fitting(params₀, ex_trace, iters; 
        n_particles=1000, n_trajs=100, trust_thresholds=(0.25, 0.75), plot_freq=10)
    ex_data = get_retval(ex_trace)
    params_est = params₀
    last_perf = nothing

    params_history = [params_est]
    traj_history = []
    score_history = []
    λ_history = []
    λ = 1.0
    @showprogress for i in 1:iters
        apf = make_particle_filter(n_particles, car_motion_model(params_est), times)
        particles, _ = smooth(apf, n_trajs, 
            columns(ex_data.controls), columns(ex_data.observations))
        fit_r = fit_dynamics(particles, times, ex_data, T, params_est)
        trajectories = rows(particles)
        perf = let 
            dl = data_likelihood.(trajectories, Ref(obs_model), Ref(columns(ex_data.observations)))
            sample_performance(0.0, 0.0, dl)
        end
        perf2 = let 
            particles2, _ = smooth(apf, n_trajs, 
                columns(ex_data.controls), columns(ex_data.observations))
            trajectories2 = rows(particles2)
            dl = data_likelihood.(trajectories2, Ref(obs_model), Ref(columns(ex_data.observations)))
            sample_performance(0.0, 0.0, dl)
        end
        if last_perf !== nothing
            improve_actual = perf - last_perf
            improve_pred = fit_r.f_final - fit_r.f_init
            perfΔ = abs(perf2 - perf)
            ρ = improve_actual / improve_pred
            @info "optimization finished." ρ improve_actual improve_pred
            if perfΔ > 0.2 * improve_pred
                @warn "High performance variance" perfΔ
            end
            if ρ > trust_thresholds[2]
                λ /= 2
            elseif ρ < trust_thresholds[1]
                λ *= 2
            end
        end
        if last_perf === nothing || ρ > 0.1
            params_est = fit_r.params
            last_perf = perf
        else
            @info("Failed to improve the objective, reject change.")
        end
        push!(params_history, params_est)
        push!(traj_history, trajectories)
        push!(score_history, perf)
        push!(λ_history, λ)
        @show λ
        @show perf
        @show params_est
        if i % plot_freq == 1
            plot_particles(particles, times, "iteration $i", ex_trace) |> display
        end
    end
    (; params_history, traj_history, score_history)
end

function plot_result((; params_history, particle_history, score_history);
        truth_trace = nothing, max_particles_to_plot=100)
    for (i, particles) in enumerate(particle_history)
        local traces = particles[1:min(length(particles), max_particles_to_plot)]
        local ts = get_args(traces[1])[2]
        plot_traces(traces, ts, "iteration $i", truth_trace) |> display
    end
    plot(score_history, title="score_history", ylabel="log_avg_prob") |> display
end

##-----------------------------------------------------------
# perform iterative synthesis
params = (; drag = 0.2, mass=1.5, σ_pos=0.1, σ_pos′=0.2)
bad_params = (; drag = 1.0, mass=5.0, σ_pos=0.5, σ_pos′=0.8)
# bad_params = (; drag = 0.254553, mass= 1.42392, σ_pos=0.31645, σ_pos′=0.584295)
syn_result = iterative_dyn_fitting(bad_params, ex_trace, 80, n_particles=1000, n_trajs=800)
show(DataFrame(syn_result.params_history), allrows=true)
# plot_result(syn_result, truth_trace=ex_trace)
##-----------------------------------------------------------