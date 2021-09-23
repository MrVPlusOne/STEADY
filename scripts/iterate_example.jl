##
using Gen
using Plots
using ProgressMeter

##
@gen function data_process(gen_mode,
        times, motion_model, obs_model, controller, controls)
    local states
    T = length(times)
    Num = Float64
    state = @trace broadcasted_normal([0.0, 0.0], [0.01, 0.5]) (:state, 1)
    if gen_mode
        states = Matrix{Num}(undef, 2, T)
        states[:, 1] = state
        controls = Vector{Num}(undef, T)
    end

    for i in 1:T-1
        Δt = times[i+1]-times[i]
        obs_distr = obs_model(state)
        obs = @trace broadcasted_normal(obs_distr...) (:obs, i)
        if gen_mode
            controls[i] = controller(state, obs)
        end
        state_distr = motion_model(state, controls[i], Δt)
        state = @trace broadcasted_normal(state_distr...) (:state, i+1)
        gen_mode && (states[:, i+1] = state)
    end
    gen_mode && return (;states, controls)
end

function states_likelihood(times, motion_model, states, controls)
    T = length(times)
    s = 0.0
    for i in 1:T-1
        state = states[i]
        Δt = times[i+1]-times[i]
        state_distr = MvNormal(motion_model(state, controls[i], Δt)...)
        s += logpdf(state_distr, states[i+1])
    end
    s    
end

function get_states(trace, T)
    cmap = get_choices(trace)
    [cmap[(:state, i)] for i in 1:T]
end
plot_trace(trace, times, name) = plot_trace!(plot(), trace, times, name)
plot_trace!(p, trace, times, name) = let
    cmap = get_choices(trace)
    states = (cmap[(:state, i)] for i in 1:length(times))
    plot!(p, times, reduce(hcat, states)', label=["x ($name)" "v ($name)"])
end

plot_traces(traces, times, name, truth_trace=nothing) = let
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

function car_motion_model(state, ctrl, Δt, (; drag, mass, σ_pos, σ_pos′))
    (pos, pos′) = state
    f = ctrl
    (; pos′′) = Car1D.acceleration_f((;f, drag, mass, pos′))
    μ = [pos + Δt * pos′,  pos′ + Δt * pos′′]
    (μ, abs.([σ_pos * Δt, σ_pos′ * Δt]))
end

function obs_model(state)
    (pos, pos′) = state
    dis = 10.0 - pos
    v_mea = pos′
    ([dis, v_mea], [1.0, 0.3])
end
#-----------------------------------------------------------
## generate data
times = collect(0.0:0.1:10.0)
T = length(times)
params = (; drag = 0.2, mass=1.5, σ_pos=0.1, σ_pos′=0.2)
controller = (s, z) -> begin
    Car1D.controller((speed=z[2], sensor=z[1])).f
end
params_to_model(params) = (s, u, Δt) -> car_motion_model(s, u, Δt, params)
ground_motion_model = params_to_model(params)
ex_trace, p_data = generate(data_process, 
    (true, times, 
    ground_motion_model,
    obs_model,
    controller,
    nothing),
)
ex_data = get_retval(ex_trace)
plot_trace(ex_trace, times, "truth") |> display
#----------------------------------------------------------- 
## randomly guess the states
obs_map = let
    content = (((:obs, i), ex_trace[(:obs, i)]) for i in 1:T-1)
    cmap = choicemap(content...)
end
post_args = (false, times, 
    (s, u, Δt) -> car_motion_model(s, u, Δt, params),
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
#-----------------------------------------------------------
## particle filter inference
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

pf_s = PF_inference_gen((;times, obs_map, ex_data.controls), ground_motion_model, 1000)
plot_traces(pf_s.traces, times, "particle filter", ex_trace)
##-----------------------------------------------------------
# performs important resampling
posterior_samples_importance(samples_per_trace, n_traces) = let
    @showprogress map(1:n_traces) do _
        trace, _ = importance_resampling(
            data_process, post_args, obs_map, samples_per_trace)
        trace
    end
end

ir_traces = posterior_samples_importance(2000, 10)
plot_traces(ir_traces, times, "IR", ex_trace)
#-----------------------------------------------------------
## dynamics fitting
to_params(vec) = let
    local drag, mass, σ_pos, σ_pos′ = vec
    (; drag, mass, σ_pos, σ_pos′)
end

to_vec((; drag, mass, σ_pos, σ_pos′)) = [drag, mass, σ_pos, σ_pos′]

function fit_dynamics(traces, times, controls, T, params_guess)
    trajectories = get_states.(traces, T)

    function loss(vec)
        local params = to_params(vec)
        s_σ = logpdf(Normal(0.1, 0.01), params.σ_pos) + logpdf(Normal(0.2, 0.001), params.σ_pos′)
        scores = map(trajectories) do states
            states_likelihood(times, 
                (s, u, Δt) -> car_motion_model(s, u, Δt, params), states, controls)
        end
        -Distributions.logsumexp(scores) - s_σ
    end

    alg = Optim.LBFGS()
    sol = Optim.optimize(loss, to_vec(params_guess), alg, autodiff=:forward)
    Optim.converged(sol) || display(sol)
    Optim.minimizer(sol) |> to_params
end

plot_trace(pf_s.traces[1], times, "particle 1")
map_params = fit_dynamics(pf_s.traces[1:50], times, ex_data.controls, T, 
    (; drag = 0.2, mass=1.0, σ_pos=0.2, σ_pos′=0.4))
#-----------------------------------------------------------
## show some rollouts
let 
    rollouts = [generate(data_process, 
        (true, times, 
        (s, u, Δt) -> car_motion_model(s, u, Δt, map_params),
        obs_model,
        controller,
        ex_data.controls),
    )[1] for _ in 1:20]
    plot_traces(rollouts, times, "map rollouts", ex_trace) |> display
end
##-----------------------------------------------------------
# now, repeat the experiment with bad parameters
function iterative_dyn_fitting(params₀, ex_data, iters)
    params_est = params₀
    params_history = [params_est]
    particle_history = []
    for i in 1:iters
        @info "iteration $i..."
        local pf_s = PF_inference_gen((;times, obs_map, ex_data.controls), 
            params_to_model(params_est), 1000)
        particles = pf_s.traces[1:200]
        params_est = fit_dynamics(particles, times, ex_data.controls, T, params_est)
        push!(params_history, params_est)
        push!(particle_history, particles)
        @show params_est
    end
    (; params_history, particle_history)
end
bad_params = (; drag = 0.8, mass=4.0, σ_pos=0.5, σ_pos′=0.8)
syn_result = iterative_dyn_fitting(bad_params, ex_data, 20)
display(syn_result.params_history)
##-----------------------------------------------------------