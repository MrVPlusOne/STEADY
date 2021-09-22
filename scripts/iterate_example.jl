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

## particle filter inference

##

function get_states(trace, T)
    cmap = get_choices(trace)
    [cmap[(:state, i)] for i in 1:T]
end

plot_trace(trace, times; title) = let
    cmap = get_choices(trace)
    states = (cmap[(:state, i)] for i in 1:length(times))
    plot(times, reduce(hcat, states)', label=["x" "v"]; title)
end

plot_traces(traces, times; title) = let
    cmaps = get_choices.(traces)
    @unzip y_mean, y_std = map(1:length(times)) do i
        ys = (x -> x[(:state, i)]).(cmaps)
        mean(ys), std(ys)
    end
    to_data(vecs) = reduce(hcat, vecs)'
    plot(times, to_data(y_mean), ribbon=to_data.((y_std, y_std)), label=["x" "v"]; title)
end

function car_motion_model(state, ctrl, Δt, (; drag, mass, σ_pos, σ_pos′))
    (pos, pos′) = state
    f = ctrl
    (; pos′′) = Car1D.acceleration_f((;f, drag, mass, pos′))
    μ = [pos + Δt * pos′,  pos′ + Δt * pos′′]
    (μ, [σ_pos * √Δt, σ_pos′ * √Δt])
end

function obs_model(state)
    (pos, pos′) = state
    dis = 10.0 - pos
    v_mea = pos′
    ([dis, v_mea], [1.0, 0.3])
end
##
times = collect(0.0:0.1:10.0)
T = length(times)
params = (; drag = 0.2, mass=1.5, σ_pos=0.1, σ_pos′=0.2)
controller = (s, z) -> begin
    @show (s, z)
    Car1D.controller((speed=z[2], sensor=z[1])).f
end
ex_trace, p_data = generate(data_process, 
    (true, times, 
    (s, u, Δt) -> car_motion_model(s, u, Δt, params),
    obs_model,
    controller,
    nothing),
)
ex_data = get_retval(ex_trace)
plot_trace(ex_trace, times, title="truth") |> display
## infer the trajectory using the ground-truth motion model
constraints = let
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
    constraints,
)
plot_trace(guessed_trace, times, title="guess") |> display

(p_data, p_guess)
hidden_variables = Gen.select(((:state, i) for i in 1:T-1)...)
## performs posterior sampling
posterior_samples_importance(samples_per_trace, n_traces) = let
    @showprogress map(1:n_traces) do _
        trace, _ = importance_resampling(
            data_process, post_args, constraints, samples_per_trace)
        trace
    end
end

posterior_traces = posterior_samples_importance(2000, 10)
plot_traces(posterior_traces, times, title="importance resampling") |> display
get_score.(posterior_traces)

map_trace = map_optimize(posterior_traces[1], hidden_variables,
    max_step_size=0.4, tau=0.5, min_step_size=1e-16, verbose=false)
get_score(map_trace)
## dynamics fitting
to_params(vec) = let
    local drag, mass, σ_pos, σ_pos′ = vec
    (; drag, mass, σ_pos, σ_pos′)
end

to_vec((; drag, mass, σ_pos, σ_pos′)) = [drag, mass, σ_pos, σ_pos′]

function fit_dynamics(traces, times, controls, T)
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

    init_guess = (; drag = 0.2, mass=1.0, σ_pos=0.2, σ_pos′=0.4)
    alg = Optim.LBFGS()
    sol = Optim.optimize(loss, to_vec(init_guess), alg, autodiff=:forward)
    display(sol)
    Optim.minimizer(sol) |> to_params
end

plot_trace(posterior_traces[2], times, title="post 1")
map_params = fit_dynamics(posterior_traces, times, ex_data.controls, T)
states_likelihood(times, (s, u, Δt) -> car_motion_model(s, u, Δt, map_params), 
    get_states(posterior_traces[1], T), ex_data.controls)
map_params
##
let 
    rollouts = [generate(data_process, 
        (true, times, 
        (s, u, Δt) -> car_motion_model(s, u, Δt, map_params),
        obs_model,
        controller,
        nothing),
    )[1] for _ in 1:20]
    plot_traces(rollouts, times, title="map rollouts") |> display
end

##
