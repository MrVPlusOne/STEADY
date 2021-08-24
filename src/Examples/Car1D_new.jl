module Car1D

using UnPack
using Distributions
using ..SEDL
using ..SEDL: TimeSeries
using StatsPlots
using DataFrames

Pos = Var(:pos, ℝ, PUnits.Length)
Power = Var(:f, ℝ, PUnits.Force)
Drag = Var(:drag, ℝ, PUnits.Force / PUnits.Speed)
Mass = Var(:mass, ℝ, PUnits.Mass)
Wall = Var(:wall, ℝ, PUnits.Length)

state_vars() = [Pos]
action_vars() = [Power]
param_vars() = [Drag]

variable_data() = VariableData(
    states = Dict(
        Pos => (Normal(0.0, 0.01), Normal(0.0, 1.0)),
    ),
    dynamics_params = Dict(
        Mass => Uniform(0.5, 5.0),
        Drag => Uniform(0.0, 1.0),
    ),
    others = Dict(
        Wall => Normal(25.0, 25.0),
    ),
)

acceleration_f(args) = begin
    @unpack f, drag, mass, pos′ = args
    (f - drag * pos′) / mass
end

sensor_max_range = 5.0
function sensor_dist(s, others; noise_scale)
    Normal(others.wall - s.pos, 0.2noise_scale)
end

function speed_dist(s; noise_scale)
    Normal(s.pos′, 0.2noise_scale)
end

function odometry_dist(s, s1; noise_scale)
    Δ = s1.pos - s.pos
    Normal(Δ, 0.1noise_scale * (1+abs(Δ)))
end

function controller(obs)
    @unpack speed, sensor = obs
    stop_dis = 2.0
    max_force = 10.0
    is_stopping = sensor < stop_dis
    target_v = is_stopping ? 0.0 : 2.0
    k = is_stopping ? 5.0 : 1.0
    (; f = clamp((target_v - speed) * k, -max_force, max_force))
end

function observe(s, s_prev, others; noise_scale)
    (
        sensor = rand(sensor_dist(s, others; noise_scale)),
        speed = rand(speed_dist(s; noise_scale)),
        odometry = rand(odometry_dist(s_prev, s; noise_scale)),
    )
end

function observe_logp(obs, s, s_prev, others; noise_scale)
    +(
        logpdf(sensor_dist(s, others; noise_scale), obs.sensor),
        logpdf(speed_dist(s; noise_scale), obs.speed),
        logpdf(odometry_dist(s_prev, s; noise_scale), obs.odometry),
    )
end

function generate_data(vdata::VariableData, times::TimeSeries; noise_scale)
    @unpack x₀, x′₀, params, others = rand(vdata)
    f_x′′ = (acceleration_f,)
    generate_data(x₀, x′₀, f_x′′, params, others, times; noise_scale)
end

function generate_data(
    x₀::NamedTuple,
    x′₀::NamedTuple,
    f_x′′::Tuple,
    params::NamedTuple,
    others::NamedTuple,
    times::TimeSeries; 
    noise_scale,
)
    # Refs to store the current time step, state, and observation
    i_ref = Ref(1)
    s_ref = Ref(merge(x₀, x′₀))
    obs_ref = Ref(observe(s_ref[], s_ref[], others; noise_scale))

    states = NamedTuple[]
    observations = NamedTuple[]
    actions = NamedTuple[]
    
    should_stop() = i_ref[] > length(times)
    
    next_time_action!() = begin
        i = i_ref[]
        i_ref[] += 1
        act = controller(obs_ref[])
        push!(actions, act)
        times[i], act
    end

    record_state!(s) = begin
        s_prev = s_ref[]
        s_ref[] = s
        obs_ref[] = observe(s, s_prev, others; noise_scale)
        push!(states, s_ref[])
        push!(observations, obs_ref[])
    end

    SEDL.simulate(x₀, x′₀, f_x′′, params, should_stop, next_time_action!, record_state!)
    (;params, others, states, observations, actions, times)
end

function data_likelihood(states, others, observations; noise_scale)
    sum(let 
        s = states[i]
        s_prev = (i == 1) ? s : states[i-1]
        observe_logp(observations[i], s, s_prev, others; noise_scale)
    end for i in 1:length(states))
end

"""
Plot the result returned by `generate_data`.
"""
function plot_data(data, name::String)
    @unpack params, others, states, observations, actions, times = data
    xs = times
    p_state = @df DataFrame(states) plot(xs, [:pos :pos′], title="States ($name)")
    hline!(p_state, [others.wall], label="wall_x")
    p_action = @df DataFrame(actions) plot(xs, :f , title="Actions ($name)")
    p_obs = @df DataFrame(observations) plot(
        xs, [:sensor :speed :odometry], title="Observations ($name)")
    plot(p_state, p_action, p_obs, layout=(3,1), size=(600,600))
end

end