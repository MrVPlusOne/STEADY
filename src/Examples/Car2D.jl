module Car2D

using Distributions
using ..SEDL
using ..SEDL: ℝ, ℝ2, °, norm_R2, sigmoid, get_rows
using ..SEDL: TimeSeries, SNormal, SMvNormal, SUniform, GDistr, landmark_readings
using StatsPlots
using StaticArrays
using Interpolations: LinearInterpolation

const Pos = Var(:pos, ℝ2, PUnits.Length)
const Orientation = Var(:θ, ℝ, PUnits.Angle)
const Vel = Var(:v, ℝ2, PUnits.Speed)
const AngularVel = Var(:ω, ℝ, PUnits.AngularSpeed)

const U_Speed = Var(:v̂, ℝ, PUnits.Speed)
const U_Turn = Var(:ϕ̂, ℝ, PUnits.Angle)

const Loc_V = Var(:loc_v, ℝ, PUnits.Speed)
const Loc_Vx = Var(:loc_vx, ℝ, PUnits.Speed)
const Loc_Vy = Var(:loc_vy, ℝ, PUnits.Speed)

variable_data(x0) = VariableData(;
    states = OrderedDict{Var, GDistr}(
        Pos => SMvNormal(x0.pos, 0.5),
        Orientation => CircularNormal(x0.θ, 15°),
        Vel => SMvNormal(x0.v, @SVector[0.5, 0.01]),
    ),
    action_vars = [U_Speed, U_Turn],
)

"""
Return the ranges and bearings to the landmarks, as well as the speed reading.
"""
function sensor_dist(landmarks; noise_scale=1.0) 
    sensor_range=7.0
    state -> begin 
        (; pos, θ, v) = state
        # v_forward = rotate2d(-θ, v)[1]
        v_forward = v[1]
        DistrIterator((
            speed = Normal(v_forward, (0.2 + 0.1norm_R2(v)) * noise_scale),
            landmarks = landmark_readings(state, landmarks; 
                sensor_range, σ_range = noise_scale * 1.0, σ_bearing = noise_scale * 5°),
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
        (t=15.0, v̂=1.5, ϕ̂=0.0),
    ]
    v̂_f = LinearInterpolation(times, v̂_seq)
    ϕ̂_f = LinearInterpolation(times, ϕ̂_seq)
    (s, obs, t::Float64) -> begin
        (v̂ = v̂_f(t), ϕ̂ = ϕ̂_f(t))
    end
end

"""
Infer angular motion, assuming no sliding.
"""
angle_only_model((; σ_pos, σ_v)) = (
    sketch = let
        param_vars = (
            l1 = Var(:l1, ℝ, PUnits.Length),
            τ1 = Var(:τ1, ℝ, PUnits.Time),
            σ_θ = Var(:σ_θ, ℝ, PUnits.unitless),
        )
    
        params = OrderedDict{Var, GDistr}(
            param_vars.l1 => PertBeta(0.2, 0.55, 0.8),
            param_vars.τ1 => PertBeta(0.01, 1.0, 10.0),
            param_vars.σ_θ => PertBeta(0.01, 5°, 60°),
        )
    
        state_to_inputs(state, ctrl) = begin
            (; pos, θ, v) = state
            (; v̂, ϕ̂) = ctrl
            loc_v = v
            (; loc_vx=loc_v[1], θ, v̂, ϕ̂)
        end
    
        outputs_to_state_dist((; der_θ), others, Δt) = begin
            (; pos, θ, v, σ_θ, τ1, v̂) = others
            loc_vx = v[1]
            
            σ_pos_scaled = (norm_R2(v) .+ 0.1) .* (σ_pos * Δt)
            dpos_dist = SMvNormal(@SVector[v[1] * Δt, 0.], σ_pos_scaled)
            der_vx = (v̂ - loc_vx) / τ1
            dv_dist = SMvNormal(@SVector[der_vx * Δt, 0.], σ_v * Δt)
            @assert σ_θ * Δt >= zero(σ_θ) "σ_θ = $σ_θ"
            DistrIterator((
                pos = pos + rotate2d(θ, dpos_dist),
                θ = Normal(θ + der_θ * Δt, σ_θ * Δt),
                v = v + dv_dist,
            ))
        end
    
        DynamicsSketch(;
            inputs = [Loc_Vx, Orientation, U_Speed, U_Turn],
            outputs = [
                Var(:der_θ, ℝ, PUnits.AngularSpeed)],
            params = params,
            state_to_inputs,
            outputs_to_state_dist,
        )
    end,
    sketch_core = input -> let
        (; loc_vx, θ, v̂, ϕ̂, τ1, l1) = input
        der_θ = tan(ϕ̂) * loc_vx / l1
        (; der_θ)
    end
)

"""
No sliding.
"""
simple_model((; σ_pos)) = (
    sketch = let
        param_vars = (
            l1 = Var(:l1, ℝ, PUnits.Length),
            τ1 = Var(:τ1, ℝ, PUnits.Time),
            σ_θ = Var(:σ_θ, ℝ, PUnits.unitless),
            σ_v = Var(:σ_v, ℝ, PUnits.Speed),
        )
    
        params = OrderedDict{Var, GDistr}(
            param_vars.l1 => PertBeta(0.2, 0.55, 0.8),
            param_vars.τ1 => PertBeta(0.01, 1.0, 10.0),
            param_vars.σ_θ => Exponential(5°),
            param_vars.σ_v => Exponential(0.2),
        )
    
        state_to_inputs(state, ctrl) = begin
            (; pos, θ, v) = state
            (; v̂, ϕ̂) = ctrl
            loc_v = v
            (; loc_vx=loc_v[1], θ, v̂, ϕ̂)
        end
    
        outputs_to_state_dist((; der_vx, der_θ), others, Δt) = begin
            (; pos, θ, v, σ_v, σ_θ) = others
            
            σ_pos_scaled = (norm_R2(v) .+ 0.1) .* (σ_pos * Δt)
            dpos_dist = SMvNormal(@SVector[v[1] * Δt, 0.], σ_pos_scaled)
            dv_dist = SMvNormal(@SVector[der_vx * Δt, 0.], σ_v * Δt)
            DistrIterator((
                pos = pos + rotate2d(θ, dpos_dist),
                θ = Normal(θ + der_θ * Δt, σ_θ * Δt),
                v = v + dv_dist,
            ))
        end
    
        DynamicsSketch(;
            inputs = [Loc_Vx, Orientation, U_Speed, U_Turn],
            outputs = [
                Var(:der_vx, ℝ, PUnits.Acceleration), 
                Var(:der_θ, ℝ, PUnits.AngularSpeed)],
            params = params,
            state_to_inputs,
            outputs_to_state_dist,
        )
    end,
    sketch_core = input -> let
        (; loc_vx, θ, v̂, ϕ̂, τ1, l1) = input
        der_vx = (v̂ - loc_vx) / τ1
        der_θ = tan(ϕ̂) * loc_vx / l1
        (; der_vx, der_θ)
    end
)

sliding_model(x0, (; σ_pos, σ_θ, σ_v)) = (
    vdata = VariableData(;
        states = OrderedDict{Var, GDistr}(
            Pos => SMvNormal(x0.pos, 5.),
            Vel => SMvNormal(x0.v, @SVector[0.5, 0.01]),
            Orientation => CircularNormal(x0.θ, 15°),
            AngularVel => CircularNormal(x0.ω, 15°),
        ),
        action_vars = [U_Speed, U_Turn],
    ),
    sketch = let
        param_vars = (
            l1 = Var(:l1, ℝ, PUnits.Length),
            τ1 = Var(:τ1, ℝ, PUnits.Time),
            a1 = Var(:a1, ℝ, PUnits.Acceleration), # linear sliding fraction
            a2 = Var(:a2, ℝ, PUnits.Acceleration), # rotational sliding fraction
            σ_θ = Var(:σ_θ, ℝ, PUnits.unitless),
            σ_v = Var(:σ_v, ℝ, PUnits.Speed),
        )
    
        params = OrderedDict{Var, GDistr}(
            param_vars.l1 => PertBeta(0.2, 0.55, 0.8),
            param_vars.τ1 => PertBeta(0.01, 1.0, 10.0),
            param_vars.a1 => PertBeta(0.5, 1.0, 5.0),
            param_vars.a2 => PertBeta(1.0, 2.0, 10.0),
            param_vars.σ_θ => Exponential(5°),
            param_vars.σ_v => Exponential(0.2),
        )
    
        state_to_inputs(state, ctrl) = begin
            (; pos, v, θ, ω) = state
            (; v̂, ϕ̂) = ctrl
            loc_v = rotate2d(-θ, v)
            (; loc_vx=loc_v[1], loc_vy=loc_v[2], θ, ω, v̂, ϕ̂)
        end
    
        outputs_to_state_dist((; der_vx, der_vy, der_θ, der_ω ), others, Δt) = begin
            (; pos, θ, v) = others
            
            σ_pos_scaled = (norm_R2(v) .+ 0.1) .* (σ_pos * Δt)
            dpos_dist = SMvNormal(@SVector[0.,0.], σ_pos_scaled)
            dv_dist = SMvNormal(@SVector[der_vx * Δt, der_vy * Δt], σ_v * Δt)
            DistrIterator((
                pos = (pos + v * Δt) + rotate2d(θ, dpos_dist),
                θ = Normal(θ + der_θ * Δt, σ_θ * Δt),
                v = v + rotate2d(π/2, v * der_θ) + rotate2d(θ, dv_dist),
            ))
        end
    
        DynamicsSketch(;
            inputs = [Loc_Vx, Loc_Vy, Orientation, U_Speed, U_Turn],
            outputs = [
                Var(:der_vx, ℝ, PUnits.Acceleration), 
                Var(:der_vy, ℝ, PUnits.Acceleration), 
                Var(:der_θ, ℝ, PUnits.AngularSpeed)],
            params = params,
            state_to_inputs,
            outputs_to_state_dist,
        )
    end,
    sketch_core = input -> let
        (; loc_vx, loc_vy, θ, v̂, ϕ̂, τ1, l1, v1, a1) = input
        der_vx = (v̂ - loc_vx) / τ1
        der_θ = tan(ϕ̂) * loc_vx / l1
        der_vy = -a1 * tanh(loc_vy / v1)
        (; der_vx, der_vy, der_θ)
    end
)




end # module