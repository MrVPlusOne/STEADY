abstract type CarDynamics end

"""
The 2D Car with sliding scenario
"""
@kwdef struct Car2dScenario{LI<:LandmarkInfo} <: Scenario
    landmark_info::LI
    car_dynamics::CarDynamics
end

@kwdef struct BicycleCarDyn <: CarDynamics
    front_drive::Bool = true
end

function Base.summary(io::IO, sce::Car2dScenario)
    print(io, "Car2d(front_drive=$(sce.car_dynamics.front_drive))")
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
    :loc_ax => ℝ(PUnits.Acceleration),
    :loc_ay => ℝ(PUnits.Acceleration),
    :der_ω => ℝ(PUnits.AngularSpeed / PUnits.Time),
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
    VariableData([pos, vel, θ, ω], [v̂, steer])
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
        lmr = landmark_readings(state, sce.landmark_info)
        DistrIterator((landmarks=lmr,))
    end
end

function car2d_inputs_transform((; pos, vel, θ, ω), (; v̂, steer))
    local loc_v = rotate2d(-θ, vel)
    (; loc_vx=loc_v[1], loc_vy=loc_v[2], ω, θ, v̂, steer)
end

function sindy_sketch(sce::Car2dScenario)
    (; loc_vx, loc_vy, ω, θ, v̂, steer, loc_ax, loc_ay, der_ω) = variables(sce)
    input_vars = [loc_vx, loc_vy, ω, θ, v̂, steer]
    output_vars = [loc_ax, loc_ay, der_ω]

    MotionModelSketch(
        input_vars,
        output_vars,
        car2d_inputs_transform,
        hover_outputs_transform,
        hover_outputs_inv_transform,
    )
end

function dynamics_sketch(sce::Car2dScenario)
    vars = variables(sce)
    inputs = [vars.loc_vx, vars.loc_vy, vars.ω, vars.steer, vars.v̂]
    outputs = [vars.f_x, vars.f_y, vars.f_θ]

    params = OrderedDict{Var,GDistr}(
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
        (; pos, vel, θ, ω, mass, rot_mass, σ_v, σ_ω) = others

        loc_ax = (f_x) / mass
        loc_ay = f_y / mass
        der_ω = (f_θ) / rot_mass
        acc = rotate2d(θ, @SVector([loc_ax, loc_ay]))
        DistrIterator((
            pos=Dirac(pos + vel * Δt),
            vel=SMvNormal(vel + acc * Δt, σ_v * Δt),
            θ=Dirac(θ + ω * Δt),
            ω=Normal(ω + der_ω * Δt, σ_ω * Δt),
        ))
    end

    DynamicsSketch(; inputs, outputs, params, state_to_inputs, outputs_to_state_dist)
end

dynamics_core(sce::Car2dScenario) = dynamics_core(sce.car_dynamics)
function dynamics_core(dyn::BicycleCarDyn)
    if dyn.front_drive
        input -> let
            (; loc_vx, loc_vy, ω, steer) = input
            (; fraction_max, v̂, mass, drag_x, drag_y, rot_mass, rot_drag, len) = input

            # assume that the tires have negligible inertia
            # and that only the front tires are driven
            v_tire_front = rotate2d(steer, @SVector([v̂, 0]))
            v_loc_front = @SVector([loc_vx, loc_vy + len * ω])
            fraction_front = unit_R2(v_tire_front - v_loc_front) * fraction_max

            vy_tire_rear = loc_vy - len * ω
            fraction_rear = -sign(vy_tire_rear) * fraction_max

            f_x = fraction_front[1]
            f_y = fraction_front[2] + fraction_rear
            f_θ = (fraction_front[2] - fraction_rear) * len
            (; f_x, f_y, f_θ)
        end
    else
        input -> let
            (; loc_vx, loc_vy, ω, steer, v̂) = input
            (; fraction_max, mass, drag_x, drag_y, rot_mass, rot_drag, len) = input

            # assume that the tires have negligible inertia
            # and that only the rear tires are driven
            v_tire_rear = @SVector([v̂, 0])
            v_loc_rear = @SVector([loc_vx, loc_vy - len * ω])
            fraction_rear = unit_R2(v_tire_rear - v_loc_rear) * fraction_max

            front_fraction_dir = rotate2d(steer, @SVector([0.0, 1.0]))
            v_loc_front = @SVector([loc_vx, loc_vy + len * ω])
            fraction_front =
                -fraction_max * unit_R2(project_R2(v_loc_front, front_fraction_dir))

            f_x = fraction_front[1] + fraction_rear[1]
            f_y = fraction_front[2] + fraction_rear[2]
            f_θ = (fraction_front[2] - fraction_rear[2]) * len
            (; f_x, f_y, f_θ)
        end
    end
end

function batched_sketch(::Car2dScenario)
    BatchedMotionSketch(;
        state_vars=(; pos=2, vel=2, θ=1, ω=1),
        control_vars=(; v̂=1, steer=1),
        input_vars=(; loc_v=2, ω=1, θ=1, v̂=1, steer=1),
        output_vars=(; loc_acc=2, a_θ=1),
        state_to_input=state_to_input_SE2,
        output_to_state=output_to_state_SE2,
        output_from_state=output_from_state_SE2,
    )
end

batched_core(sce::Car2dScenario, params) = batched_core(sce.car_dynamics, params)
function batched_core(dyn::BicycleCarDyn, params)
    if dyn.front_drive
        # rear drive
        (input::BatchTuple) -> let
            (; tconf, batch_size) = input
            (; loc_v, ω, steer, v̂) = input.val
            (; fraction_max, mass, rot_mass, len, σ_v, σ_ω) = params

            # assume that the tires have negligible inertia
            # and that only the rear tires are driven
            v_tire_front =
                vcat(cos.(steer) .* v̂, sin.(steer) .* v̂ .- len .* ω) - loc_v

            fraction_front = unit_R2(v_tire_front) * fraction_max

            vy_tire_rear = @views loc_v[2:2, :] - len * ω
            fraction_rear = -sign.(vy_tire_rear) .* fraction_max

            loc_acc = (fraction_front + vcat(zero(fraction_rear), fraction_rear)) / mass
            a_θ = @views (fraction_front[2:2, :] - fraction_rear) * len / rot_mass

            μs = BatchTuple(tconf, batch_size, (; loc_acc, a_θ))
            σs = BatchTuple(tconf, batch_size, (loc_acc=tconf([σ_v;;]), a_θ=tconf([σ_ω;;])))
            (; μs, σs)
        end
    else
        # rear drive
        (input::BatchTuple) -> let
            (; tconf, batch_size) = input
            (; loc_v, ω, steer, v̂) = input.val
            (; fraction_max, mass, rot_mass, len, σ_v, σ_ω) = params

            # assume that the tires have negligible inertia
            # and that only the rear tires are driven
            v_tire_rear = vcat_bc(v̂, len * ω; batch_size) - loc_v
            fraction_rear = unit_R2(v_tire_rear) * fraction_max

            front_fraction_dir = [-sin.(steer); cos.(steer)]
            v_loc_front = loc_v + [zero(ω); len * ω]
            fraction_front =
                -fraction_max * unit_R2(project_R2(v_loc_front, front_fraction_dir))

            loc_acc = (fraction_front + fraction_rear) / mass
            a_θ = @views (fraction_front[2:2, :] - fraction_rear[2:2, :]) * len / rot_mass

            μs = BatchTuple(tconf, batch_size, (; loc_acc, a_θ))
            σs = BatchTuple(tconf, batch_size, (loc_acc=tconf([σ_v;;]), a_θ=tconf([σ_ω;;])))
            (; μs, σs)
        end
    end
end

function plot_scenario!(sce::Car2dScenario, states, obs_data, name; plt_args...)
    plot_2d_scenario!(states, obs_data, name; sce.landmark_info.landmarks, plt_args...)
end

function plot_trajectories!(::Car2dScenario, trajectories, name; plt_args...)
    plot_2d_trajectories!(trajectories, name; plt_args...)
end

function get_simplified_motion_model(sce::Car2dScenario, (; len, σ_v, σ_ω))
    (state::NamedTuple, ctrl::NamedTuple, Δt::Real) -> begin
        (; pos, vel, θ, ω) = state
        (; steer, v̂) = ctrl

        vel_pred = rotate2d(θ, @SVector([v̂, 0.0]))
        ω_pred = tan(steer) * v̂ / 2len

        DistrIterator((
            pos=SMvNormal(pos + vel_pred * Δt, 5max(abs(v̂), 0.1) * Δt * σ_v),
            vel=Dirac(vel_pred),
            θ=Normal(θ + ω_pred * Δt, 5max(abs(ω_pred), 0.1) * Δt * σ_ω),
            ω=Dirac(ω_pred),
        ))
    end
end

function simulation_controller(sce::Car2dScenario; noise=0.5)
    pert(x, σ) = x + noise * σ * randn()
    front_drive = sce.car_dynamics.front_drive
    @unzip times, v̂_seq, steer_seq = if front_drive
        [
            (t=0.0, v̂=0.0, steer=0.0),
            (t=1.0, v̂=pert(3.2, 1.0), steer=pert(10°, 7°)),
            (t=4.0, v̂=pert(3.0, 1.0), steer=pert(10°, 7°)),
            (t=4.5, v̂=pert(3.3, 1.0), steer=pert(-30°, 7°)),
            (t=6.5, v̂=pert(3.3, 1.0), steer=pert(-30°, 7°)),
            (t=7.2, v̂=pert(2.0, 1.0), steer=pert(20°, 7°)),
            (t=9.0, v̂=pert(1.8, 1.0), steer=pert(20°, 7°)),
            (t=9.6, v̂=pert(2.5, 1.0), steer=pert(10°, 7°)),
            (t=15.0, v̂=2.0, steer=0.0),
        ]
    else
        [
            (t=0.0, v̂=0.0, steer=0.0),
            (t=1.0, v̂=pert(2.0, 1.0), steer=pert(0°, 7°)),
            (t=4.0, v̂=pert(2.0, 1.0), steer=pert(0°, 7°)),
            (t=4.5, v̂=pert(2.5, 1.0), steer=pert(-20°, 7°)),
            (t=6.5, v̂=pert(2.9, 1.0), steer=pert(-20°, 7°)),
            (t=7.2, v̂=pert(3.2, 1.0), steer=pert(20°, 7°)),
            (t=9.0, v̂=pert(3.0, 1.0), steer=pert(10°, 7°)),
            (t=10.6, v̂=pert(2.4, 1.0), steer=pert(0°, 7°)),
            (t=15.0, v̂=2.0, steer=0.0),
        ]
    end
    if rand() < 0.6
        steer_seq = -steer_seq
    end
    v̂_f = LinearInterpolation(times, v̂_seq)
    steer_f = LinearInterpolation(times, steer_seq)
    (s, obs, t::Real) -> begin
        (v̂=v̂_f(t), steer=steer_f(t))
    end
end

function simulation_x0(::Car2dScenario)
    (
        pos=@SVector([-2.5 + 3randn(), 3randn()]),
        vel=@SVector([0.3randn(), 0.3randn()]),
        θ=randn(),
        ω=10° * randn(),
    )
end

function simulation_params(::Car2dScenario)
    (;
        mass=2.0,
        drag_x=0.05,
        drag_y=0.11,
        rot_mass=0.65,
        rot_drag=0.07,
        sep=0.48,
        len=0.42,
        fraction_max=1.5,
        σ_v=0.04,
        σ_ω=0.03,
    )
end

function car2d_scenario(; front_drive=false)
    landmarks = @SVector([
        @SVector([-1.0, 2.5]),
        @SVector([6.0, -4.0]),
        @SVector([6.0, 12.0]),
        @SVector([10.0, 2.0]),
    ])
    lInfo = LandmarkInfo(; landmarks, bearing_only=Val(false))
    Car2dScenario(lInfo, BicycleCarDyn(; front_drive=true))
end

state_L2_loss(::Car2dScenario) = L2_in_SE2