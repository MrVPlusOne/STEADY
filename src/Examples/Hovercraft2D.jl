"""
The 2D hovercraft scenario
"""
@kwdef(
struct HovercraftScenario{N} <: Scenario
    landmarks::SVector{N,SVector{2, Float64}}
    noise_scale::Float64=1.0
    sensor_range::Float64=10.0
end)

variables(::HovercraftScenario) = variable_tuple(
    # state
    :pos => ℝ2(PUnits.Length),
    :vel => ℝ2(PUnits.Speed),
    :θ => ℝ(PUnits.Angle),
    :ω => ℝ(PUnits.AngularSpeed),
    # control
    :ul => ℝ(PUnits.Force),
    :ur => ℝ(PUnits.Force),
    
    # sketch
    :loc_vx => ℝ(PUnits.Speed),
    :loc_vy => ℝ(PUnits.Speed),
    :loc_ax => ℝ(PUnits.Acceleration),
    :loc_ay => ℝ(PUnits.Acceleration),
    :der_ω => ℝ(PUnits.AngularSpeed/PUnits.Time),
    :f_x => ℝ(PUnits.Force),
    :f_y => ℝ(PUnits.Force),
    :f_θ => ℝ(PUnits.Force * PUnits.Length),
    
    # params
    :mass => ℝ(PUnits.Mass),
    :drag_x => ℝ(PUnits.Force / PUnits.Speed),
    :drag_y => ℝ(PUnits.Force / PUnits.Speed),
    :rot_mass => ℝ(PUnits.Force * PUnits.Length / PUnits.AngularAcceleration),
    :rot_drag => ℝ(PUnits.Force * PUnits.Length / PUnits.AngularSpeed),
    :sep => ℝ(PUnits.Length),
    :σ_v => ℝ(PUnits.Speed),
    :σ_ω => ℝ(PUnits.AngularSpeed),
)

function variable_data(scenario::HovercraftScenario) 
    (; pos, vel, θ, ω) = variables(scenario) 
    (; ul, ur) = variables(scenario) 
    VariableData{1}(
        [pos, vel, θ, ω],
        [ul, ur],
    )
end

function initial_state_dist(::HovercraftScenario, x0)
    DistrIterator((
        pos=SMvNormal(x0.pos, 0.2), 
        vel=SMvNormal(x0.vel, 0.2), 
        θ=CircularNormal(x0.θ, 8°),
        ω=Normal(x0.ω, 0.1),
    ))
end

"""
Observations consist only of landmark range and bearing readings.
"""
function observation_dist(sce::HovercraftScenario)
    (; landmarks, noise_scale) = sce
    state -> begin
        lmr=landmark_readings(state, landmarks; bearing_only=Val{false}(),
            sce.sensor_range, σ_range = noise_scale * 1.2, σ_bearing = noise_scale * 5°)
        DistrIterator((landmarks = lmr,))
    end
end

function dynamics_sketch(sce::HovercraftScenario)
    vars = variables(sce)
    inputs = [vars.loc_vx, vars.loc_vy, vars.ω, vars.ul, vars.ur]
    outputs = [vars.f_x, vars.f_y, vars.f_θ]

    params = OrderedDict{Var, GDistr}(
        vars.mass => PertBeta(1.0, 1.45, 1.9),
        vars.drag_x => Uniform(0.0, 0.15),
        vars.drag_y => Uniform(0.0, 0.15),
        vars.rot_mass => InverseGamma(5, 5),
        vars.rot_drag => Uniform(0.0, 0.2),
        vars.sep => PertBeta(0.72, 0.8, 0.93),
        vars.σ_v => InverseGamma(11, 1),
        vars.σ_ω => InverseGamma(11, 1),
    )

    state_to_inputs(state, ctrl) = begin
        (; pos, vel, θ, ω) = state
        (; ul, ur) = ctrl
        loc_v = rotate2d(-θ, vel)
        (; loc_vx=loc_v[1], loc_vy=loc_v[2], ω, ul, ur)
    end

    outputs_to_state_dist((; f_x, f_y, f_θ), others, Δt) = begin
        # TODO: think about how to use leap-frog with Brownian disturbance

        (; pos, vel, θ, ω, ul, ur, mass, rot_mass, sep, σ_v, σ_ω) = others

        loc_ax = (ul + ur + f_x) / mass
        loc_ay = f_y / mass
        der_ω = ((ur - ul) * sep + f_θ) / rot_mass
        acc = rotate2d(θ, @SVector[loc_ax, loc_ay])
        DistrIterator((
            pos = Dirac(pos + vel * Δt),
            vel = SMvNormal(vel + acc * Δt, σ_v * Δt),
            θ = Dirac(θ + ω * Δt),
            ω = Normal(ω + der_ω * Δt, σ_ω * Δt),
        ))
    end

    DynamicsSketch(;
        inputs, 
        outputs,
        params,
        state_to_inputs,
        outputs_to_state_dist,
    )
end

function dynamics_core(::HovercraftScenario)
    input -> let
        (; loc_vx, loc_vy, ω, ul, ur, mass, drag_x, drag_y, rot_mass, rot_drag) = input
        f_x = -drag_x * loc_vx
        f_y = -drag_y * loc_vy
        f_θ = -ω * rot_drag
        (; f_x, f_y, f_θ)
    end
end

function plot_scenario!(sce::HovercraftScenario, states, obs_data, name; plt_args...)
    plot_2d_scenario!(states, obs_data, name; sce.landmarks, plt_args...)
end

function plot_trajectories!(::HovercraftScenario, trajectories, name; plt_args...)
    plot_2d_trajectories!(trajectories, name; plt_args...)
end