module Car2D

using Distributions
using ..SEDL
using ..SEDL: ℝ, ℝ2, °, norm_R2, TimeSeries, SNormal, SMvNormal, SUniform, GDistr
using StatsPlots
using DataFrames
using StaticArrays
using Interpolations: LinearInterpolation

Pos = Var(:pos, ℝ2, PUnits.Length)
Orientation = Var(:θ, ℝ, PUnits.Angle)
Vel = Var(:v, ℝ2, PUnits.Speed)

U_speed = Var(:v̂, ℝ, PUnits.Speed)
U_turn = Var(:ϕ̂, ℝ, PUnits.Angle)

ParamVars = (
    k1 = Var(:k1, ℝ, PUnits.unitless),
    k2 = Var(:k2, ℝ, PUnits.unitless),
    l = Var(:l, ℝ, PUnits.Length),
    τ1 = Var(:τ1, ℝ, PUnits.Time),
    τ2 = Var(:τ2, ℝ, PUnits.Time),
)

variable_data() = VariableData(;
    states = OrderedDict{Var, GDistr}(
        Pos => SMvNormal(@SVector[0., 0.], 0.01),
        Orientation => SNormal(0.0, 0.01),
        Vel => SMvNormal(@SVector[0., 0.], @SVector[0.5, 0.01]),
    ),
    params = OrderedDict(
        ParamVars.k1 => Exponential(1.0),
        ParamVars.k2 => Exponential(1.0),
        ParamVars.l => PertBeta(0.2, 0.8, 0.5),
        ParamVars.τ1 => Exponential(1.0),
        ParamVars.τ2 => Exponential(1.0),
    ),
    action_vars = [U_speed, U_turn],
)

"""
Return the ranges and bearings to the landmarks, as well as the speed reading.
"""
function sensor_dist(landmarks, noise_scale) 
    state -> begin 
        (; pos, θ, v) = state
        v_forward = rotate2d(-θ, v)[1]
        DistrIterator((
            speed = Normal(v_forward, (0.1 + 0.1norm_R2(v)) * noise_scale),
            landmarks = DistrIterator(map(landmarks) do l
                rel = l - pos
                dis = norm_R2(rel)
                angle = atan(rel[2], rel[1]) - θ
                DistrIterator((
                    range = Normal(dis, noise_scale * 0.25), 
                    bearing = CircularNormal(angle, noise_scale * 10°)))
            end),
        ))
    end
end

function manual_control()
    @unzip times, v̂_seq, ϕ̂_seq = [
        (t=0.0, v̂=0.0, ϕ̂=0.0),
        (t=1.0, v̂=1.5, ϕ̂=0.0),
        (t=2.0, v̂=2.0, ϕ̂=10°),
        (t=6.0, v̂=2.2, ϕ̂=-15°),
        (t=12.0, v̂=1.0, ϕ̂=-5°),
    ]
    v̂_f = LinearInterpolation(times, v̂_seq)
    ϕ̂_f = LinearInterpolation(times, ϕ̂_seq)
    (s, obs, t::Float64) -> begin
        (v̂ = v̂_f(t), ϕ̂ = ϕ̂_f(t))
    end
end

function car2d_motion_model(velocity_f, (; σ_pos, σ_θ, σ_v))
    (state, ctrl, Δt) -> begin
        (; pos, θ, v) = state
        (; v̂, ϕ̂) = ctrl
        v_car = rotate2d(-θ, v)
        (d_vx, d_vy, d_θ) = values(velocity_f((; v_car, θ, v̂, ϕ̂))) .* Δt

        v_car_dist = SMvNormal(@SVector[v_car[1] + d_vx, 0.0], σ_v * Δt)

        σ_pos_scaled = (abs.(v_car) .+ 0.1) .* (σ_pos * Δt)

        DistrIterator((
            pos = (pos + v * Δt) + rotate2d(θ, SMvNormal(@SVector[0.,0.], σ_pos_scaled)),
            θ = Normal(θ + d_θ, σ_θ * Δt),
            v = rotate2d(θ, v_car_dist),
        ))
    end
end

function car_velocity_f((; τ_v, l, fraction_vy, inertia_vy))
    state -> begin
        (; v_car, θ, v̂, ϕ̂) = state
        der_vx = (v̂ - v_car[1]) / τ_v
        der_θ = tan(ϕ̂) * v_car[1] / l
        # der_vy = -inertia_vy * der_θ * v_car[1] - sign(v_car[2]) * fraction_vy 
        # generalize this
        der_vy = 0.0
        (der_vx, der_vy, der_θ)
    end
end

function plot_states!(states, name; marker_len=0.25, marker_thining=10, linecolor=1)
    arrow_style = arrow(:closed, 0.001, 1.0)
    let
        @unzip xs, ys = map(x -> x.pos, states)
        plot!(xs, ys, arrow=arrow_style, 
            label="Position ($name)", aspect_ratio=1.0; linecolor)
    end
    let
        markers = states[1:marker_thining:end]
        @unzip xs, ys = map(x -> x.pos, markers)
        dirs = map(markers) do x
            rotate2d(x.θ, @SVector[marker_len, 0.])
        end
        @unzip us, vs = dirs
        quiver!(xs, ys, quiver=(us, vs), arrow=arrow_style, arrowsize=0.01, 
            label="Orientation ($name)"; linecolor)
    end
end

plot_particles!(particles::Matrix, name, true_states=nothing) = begin
    @unzip xs, ys = map(x -> x.pos, particles) 
    # plt = scatter!(xs, ys, label="particles ($name)", markersize=1.0, markerstrokewidth=0)
    plt = plot!(xs', ys', label=false, linecolor=2, linealpha=0.2)
    map_optional(true_states) do states
        plot_states!(states, "truth", linecolor=1)
    end
    plt
end


end # module