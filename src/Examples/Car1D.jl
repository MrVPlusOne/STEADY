module Car1D

using Distributions
using ..SEDL
using ..SEDL: ℝ, ℝ2, TimeSeries, SNormal, SMvNormal, SUniform
using StatsPlots
using DataFrames

Pos = Var(:pos, ℝ, PUnits.Length)
Power = Var(:f, ℝ, PUnits.Force)
Drag = Var(:drag, ℝ, PUnits.Force / PUnits.Speed)
Mass = Var(:mass, ℝ, PUnits.Mass)

state_vars() = [Pos]
action_vars() = [Power]
param_vars() = [Drag, Mass]

variable_data() = VariableData(
    Val(2);
    states=OrderedDict(Pos => (SNormal(0.0, 0.01), SNormal(0.0, 1.0))),
    dynamics=OrderedDict(Mass => SUniform(0.5, 5.0), Drag => SUniform(0.0, 1.0)),
    action_vars=[Power],
)

acceleration_f((; f, drag, mass, pos′)) = begin
    (pos′′=(f - drag * pos′) / mass,)
end

sensor_max_range = 5.0
function sensor_dist(s, others; noise_scale)
    SNormal(others.wall - s.pos, 0.2noise_scale)
end

function speed_dist(s; noise_scale)
    SNormal(s.pos′, 0.2noise_scale)
end

function odometry_dist(s, s1; noise_scale)
    Δ = s1.pos - s.pos
    SNormal(Δ, 0.1noise_scale * (1 + abs(Δ)))
end

function controller((; speed, sensor))
    stop_dis = 2.0
    max_force = 10.0
    is_stopping = sensor < stop_dis
    target_v = is_stopping ? 0.0 : 2.0
    k = is_stopping ? 5.0 : 1.0
    (; f=clamp((target_v - speed) * k, -max_force, max_force))
end

function obs_dist(s, s_prev, others; noise_scale)
    DistrIterator((
        sensor=sensor_dist(s, others; noise_scale),
        speed=speed_dist(s; noise_scale),
        odometry=odometry_dist(s_prev, s; noise_scale),
    ))
end

function observe(s, s_prev, others; noise_scale)
    rand(obs_dist(s, s_prev, others; noise_scale))
end

function observe_logp(obs, s, s_prev, others; noise_scale)
    logpdf(obs_dist(s, s_prev, others; noise_scale), obs)
end

generate_data(x₀, x′₀, params, others, times; noise_scale) = SEDL.generate_data(
    x₀, x′₀, acceleration_f, params, others, times; observe, controller, noise_scale
)

function data_likelihood(states, others, observations; noise_scale)
    sum(
        let
            s = states[i]
            s_prev = (i == 1) ? s : states[i - 1]
            observe_logp(observations[i], s, s_prev, others; noise_scale)
        end for i in 1:length(states)
    )
end

"""
Plot the result returned by `generate_data`.
"""
function plot_data((; others, states, observations, actions, times), name::String)
    xs = times
    p_state = @df DataFrame(states) plot(xs, [:pos :pos′], title="States ($name)")
    hline!(p_state, [others.wall]; label="wall_x")
    p_action = @df DataFrame(actions) plot(xs, :f, title="Actions ($name)")
    p_obs = @df DataFrame(observations) plot(
        xs, [:sensor :speed :odometry], title="Observations ($name)"
    )
    plot(p_state, p_action, p_obs; layout=(3, 1), size=(600, 600))
end

end # end module