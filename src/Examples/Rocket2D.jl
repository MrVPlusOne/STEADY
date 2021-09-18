module Rocket2D

using ..SEDL
using ..SEDL: ℝ, ℝ2, rotate2d, norm_R2, TimeSeries
using ..SEDL: SMvNormal, SNormal, SUniform, SMvUniform
using StatsPlots
using DataFrames
using Distributions
using StaticArrays
import Random

Pos = Var(:pos, ℝ2, PUnits.Length)
"θ is defined as the angle between the rocket head and the y axis."
Orientation = Var(:θ, ℝ, PUnits.unitless)
Thrust = Var(:thrust, ℝ, PUnits.Force)
Turn = Var(:turn, ℝ, PUnits.Force * PUnits.Length)
Drag = Var(:drag, ℝ, PUnits.Force / PUnits.Speed)
RotDrag = Var(:rot_drag, ℝ, PUnits.Force * PUnits.Length / PUnits.AngularSpeed)
Mass = Var(:mass, ℝ, PUnits.Mass)
RotMass = Var(:rot_mass, ℝ, PUnits.Mass * PUnits.Length^2)
RocketLength = Var(:length, ℝ, PUnits.Length)
Gravity = Var(:gravity, ℝ2, PUnits.Force)

state_vars() = [Pos, Orientation]
action_vars() = [Thrust, Turn]
param_vars() = [Drag, RotDrag, Mass, RotMass, RocketLength, Gravity]

landmark_dist(n) = begin
    # DistrIterator([SMvNormal([0., -10.], [30.0, 10.0]) for _ in 1:n])
    DistrIterator([SMvUniform((-40., 40.), (-20., 10.0)) for _ in 1:n])
end

variable_data(n_landmarks, (; pos, θ), x₀_σ=0.01) = VariableData(
    states = OrderedDict(
        Pos => (SMvNormal(pos, x₀_σ), SMvNormal([0.0, 0.0], 1.0)),
        Orientation => (SNormal(θ, x₀_σ), SNormal(0.0, 0.5)),
    ),
    dynamics_params = OrderedDict(
        Drag => SUniform(0.001, 1.0),
        Mass => SUniform(0.5, 5.0),
        RotMass => SUniform(0.1, 5.0),
        RotDrag => SUniform(0.001, 1.0),
        RocketLength => SUniform(0.1,1.0),
        Gravity => SMvNormal([0.0, -1.0], [0.2, 0.5]),
    ),
    others = OrderedDict(
        :landmarks => landmark_dist(n_landmarks)),
)

# variable_data(n_landmarks, (; pos, θ), x₀_σ=0.01) = VariableData(
#     states = OrderedDict(
#         Pos => (SMvNormal(pos, x₀_σ), SMvNormal([0.0, 0.0], 1.0)),
#         Orientation => (SNormal(θ, x₀_σ), SNormal(0.0, 0.5)),
#     ),
#     dynamics_params = OrderedDict(
#         Drag => SNormal(0.1, 0.5),
#         Mass => Exponential(1.0), # SUniform(0.5, 5.0),
#         RotMass => Exponential(1.0), # SUniform(0.1, 5.0),
#         RotDrag => SNormal(0.1, 0.5), # SUniform(0.001, 1.0),
#         RocketLength => Exponential(1.0), # SUniform(0.1,1.0),
#         Gravity => SMvNormal([0.0, -1.0], [0.1, 1.0]), # SMvUniform((-0.1, 0.1), (0.1, 3.0)),
#     ),
#     others = OrderedDict(
#         :landmarks => landmark_dist(n_landmarks)),
# )

limit_control((; thrust, turn)) =
    (thrust = clamp(thrust, 0, 5.0), turn=clamp(turn, -2.0, 2.0))

function acceleration_f(
        (; pos′, θ, θ′, thrust, turn, drag, rot_drag, mass, rot_mass, gravity, length))
    # speed = norm_R2(pos′)
    thrust_force = rotate2d(θ, @SVector[0., thrust])
    gravity_force = gravity * mass
    gravity_moment = -(cos(θ)*gravity[1] + sin(θ)*gravity[2]) * length/2 * mass
    drag_force = - drag * pos′
    rot_drag_moment = - rot_drag * θ′

    pos′′ = (thrust_force + drag_force + gravity_force) / mass
    θ′′ = (turn  + rot_drag_moment + gravity_moment) / rot_mass
    (; pos′′, θ′′)
end

ground_truth_sketch() = begin
    combine(inputs, _) = acceleration_f(inputs)
    
    DynamicsSketch(
        Var[],
        combine,
    )
end

sensor_max_range = 10.0
# function sensor_dist(s, landmarks; noise_scale)
#     ranges = DistrIterator([SNormal(norm_R2(s.pos - l), 0.2noise_scale) for l in landmarks])
#     bearings = DistrIterator([let 
#         d = l - s.pos
#         θ = atan(d[2], d[1])
#         SNormal(θ - s.θ, 0.1noise_scale)  # relative angle
#     end for l in landmarks])
#     DistrIterator((; ranges, bearings))
# end

# simpler version for debugging
function sensor_dist(s, landmarks; noise_scale)
    DistrIterator([SMvNormal(l, 0.2noise_scale) for l in landmarks])
end

function gps_reading(s; noise_scale)
    SMvNormal(s.pos, 1noise_scale)
end

function Δv_meter(s, s1; noise_scale)
    Δ = s1.pos′ - s.pos′
    SMvNormal(Δ, 0.2noise_scale * (1+norm_R2(Δ)))
end

function rot_meter(s; noise_scale)
    SNormal(s.θ′, 0.2noise_scale)
end

function observation_dist(s, s_prev, others; noise_scale)
    DistrIterator((
        sensor = sensor_dist(s, others.landmarks; noise_scale),
        gps = gps_reading(s; noise_scale),
        Δv = Δv_meter(s, s_prev; noise_scale),
        ω = rot_meter(s; noise_scale), 
    ))
end

Base.isfinite(v::AbstractVector) = all(isfinite.(v))

# assuming access to the ground-truth state 
function controller(state, obs; target_pos, weight::Real, rng=Random.GLOBAL_RNG)
    if !all(map(isfinite, state))
        return (thrust=0.0, turn=0.0) # to prevent DomainError.
    end

    (; pos, pos′, θ, θ′) = state
    vel, ω = pos′, θ′

    turn_factor = exp(randn(rng)*0.2)
    thrust_factor = exp(randn(rng)*0.2)

    Δpos = pos - target_pos
    turn = -(3θ/cos(θ)+ω-0.15Δpos[1]) * turn_factor
    thrust = (weight-(0.3Δpos[2]+0.2vel[2]) * thrust_factor) / cos(θ)
    (; thrust, turn) |> limit_control
end


generate_data(x₀, x′₀, params, others, times; noise_scale, target_pos) = begin
    (; mass, gravity) = params
    weight = -gravity[2] * mass
    SEDL.generate_data(x₀, x′₀, acceleration_f, params, others, times; noise_scale,
        observe = rand ∘ observation_dist, 
        controller = (s, o) -> controller(s, o; 
            weight, target_pos), allow_state_access=true)
end


function data_likelihood(states, others, observations; noise_scale, check_dual=false)
    dl(i) = let 
        s = states[i]
        s_prev = (i == 1) ? s : states[i-1]
        v = logpdf(observation_dist(s, s_prev, others; noise_scale), observations[i])
        if check_dual && SEDL.is_bad_dual(v)
            @info "data_likelihood" i s v
            @info "data_likelihood" others
            @info "data_likelihood" distr = observation_dist(s, s_prev, others; noise_scale)
            error("bad dual detected.")
        end
        v
    end
    sum(dl(i) for i in 1:length(states))
end

function plot_data!(p, ex_data, name::String; marker_len=1.0, marker_thining=10)
    arrow_style = arrow(:closed, 0.001, 1.0)
    let
        states = ex_data.states[1:marker_thining:end]
        @unzip xs, ys = map(x -> x.pos, states)
        dirs = map(states) do x
            rotate2d(x.θ, @SVector[0.0, marker_len])
        end
        @unzip us, vs = dirs
        quiver!(p, xs, ys, quiver=(us, vs), arrow=arrow_style, arrowsize=0.01, label="Orientation ($name)")
    end
    let
        @unzip xs, ys = map(x -> x.pos, ex_data.states)
        plot!(p, xs, ys, arrow=arrow_style, label="Position ($name)")
    end
    let
        @unzip xs, ys = ex_data.others.landmarks
        scatter!(p, xs, ys, label="Landmarks ($name)")
    end
    p
    # plot(traj_plot, aspect_ratio=1, title="($name)")
end

function plot_data(ex_data, name::String; kargs...)
    p = plot(; aspect_ratio=1, legend = :outerbottom, title="Rocket2D.jl")
    plot_data!(p, ex_data, name; kargs...)
end

end # end module