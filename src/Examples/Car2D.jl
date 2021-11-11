abstract type CarDynamics end

"""
The 2D Car with sliding scenario
"""
@kwdef(
struct Car2dScenario{N} <: Scenario
    landmark_info::LandmarkInfo{true, N}
    car_dynamics::CarDynamics
end)

struct SimpleSlidingCar <: CarDynamics 

end

variables(::Car2dScenario) = variable_tuple(
    # state
    :pos => ℝ2(PUnits.Length),
    :vel => ℝ2(PUnits.Speed),
    :θ => ℝ(PUnits.Angle),
    :ω => ℝ(PUnits.AngularSpeed),
    # control
    :v̂ => ℝ(PUnits.Speed), # wheel speed
    :steer => ℝ(PUnits.Angle), # steering angle

    # sketch
    :loc_vx => ℝ(PUnits.Speed),
    :loc_vy => ℝ(PUnits.Speed),
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
    :len => ℝ(PUnits.Length),
    :fraction_max => ℝ(PUnits.Force),
    :σ_v => ℝ(PUnits.Speed),
    :σ_ω => ℝ(PUnits.AngularSpeed),
)

function variable_data(scenario::Car2dScenario) 
    (; pos, vel, θ, ω) = variables(scenario) 
    (; v̂, steer) = variables(scenario) 
    VariableData(
        [pos, vel, θ, ω],
        [v̂, steer],
    )
end

function initial_state_dist(::Car2dScenario, x0)
    DistrIterator((
        pos=SMvNormal(x0.pos, 0.2), 
        vel=SMvNormal(x0.vel, 0.2), 
        θ=CircularNormal(x0.θ, 8°),
        ω=Normal(x0.ω, 0.1),
    ))
end

function observation_dist(sce::Car2dScenario)
    state -> begin
        lmr=landmark_readings(state, sce.landmark_info)
        DistrIterator((landmarks = lmr,))
    end
end

function dynamics_sketch(sce::Car2dScenario)
    vars = variables(sce)
    inputs = [vars.loc_vx, vars.loc_vy, vars.ω, vars.steer, vars.v̂]
    outputs = [vars.f_x, vars.f_y, vars.f_θ]

    params = OrderedDict{Var, GDistr}(
        vars.mass => PertBeta(1.0, 1.45, 1.9),
        vars.drag_x => Uniform(0.0, 0.15),
        vars.drag_y => Uniform(0.0, 0.15),
        vars.rot_mass => InverseGamma(5, 5),
        vars.rot_drag => Uniform(0.0, 0.2),
        vars.sep => PertBeta(0.3, 0.45, 0.8),
        vars.len => PertBeta(0.3, 0.4, 0.5),
        vars.fraction_max => PertBeta(1.0, 3.0, 10.0),
        vars.σ_v => InverseGamma(11, 1),
        vars.σ_ω => InverseGamma(11, 1),
    )

    state_to_inputs(state, ctrl) = begin
        (; pos, vel, θ, ω) = state
        (; steer, v̂) = ctrl
        loc_v = rotate2d(-θ, vel)
        (; loc_vx=loc_v[1], loc_vy=loc_v[2], ω, steer, v̂)
    end

    outputs_to_state_dist((; f_x, f_y, f_θ), others, Δt) = begin
        # TODO: think about how to use leap-frog with Brownian disturbance

        (; pos, vel, θ, ω, mass, rot_mass, σ_v, σ_ω) = others

        loc_ax = (f_x) / mass
        loc_ay = f_y / mass
        der_ω = (f_θ) / rot_mass
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

function dynamics_core(sce::Car2dScenario)
    input -> let
        (; loc_vx, loc_vy, ω, steer) = input
        (; fraction_max, v̂, mass, drag_x, drag_y, rot_mass, rot_drag, len) = input

        # assume that the tires have negligible inertia
        # and that only the front tires are driven
        v_tire_front = rotate2d(steer, @SVector[v̂, 0])
        v_loc_front = @SVector[loc_vx, loc_vy + len * ω]
        fraction_front = unit_R2(v_tire_front - v_loc_front) * fraction_max

        vy_tire_rear = loc_vy - len * ω
        fraction_rear = -sign(vy_tire_rear) * fraction_max

        f_x = fraction_front[1]
        f_y = fraction_front[2] + fraction_rear
        f_θ = (fraction_front[2] - fraction_rear) * len
        (; f_x, f_y, f_θ)
    end
end

function plot_scenario!(sce::Car2dScenario, states, obs_data, name; plt_args...)
    plot_2d_scenario!(states, obs_data, name; sce.landmark_info.landmarks, plt_args...)
end

function plot_trajectories!(::Car2dScenario, trajectories, name; plt_args...)
    plot_2d_trajectories!(trajectories, name; plt_args...)
end