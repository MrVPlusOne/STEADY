"""
The 2D hovercraft scenario.
"""
@kwdef(struct HovercraftScenario{LI<:LandmarkInfo} <: Scenario
    landmark_info::LI
end)

Base.summary(io::IO, ::HovercraftScenario) = print(io, "Hovercarft()")

dummy_state(::HovercraftScenario) =
    (pos=to_svec(randn(2)), vel=to_svec(randn(2)), θ=randn(), ω=randn())

dummy_control(::HovercraftScenario) = (ul=randn(), ur=randn())

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
    :σ_v => ℝ(PUnits.Speed),
    :σ_ω => ℝ(PUnits.AngularSpeed),
)

function variable_data(scenario::HovercraftScenario)
    (; pos, vel, θ, ω) = variables(scenario)
    (; ul, ur) = variables(scenario)
    VariableData([pos, vel, θ, ω], [ul, ur])
end

function initial_state_dist(::HovercraftScenario, x0)
    DistrIterator((
        pos=SMvNormal(x0.pos, 0.2),
        vel=SMvNormal(x0.vel, 0.2),
        θ=CircularNormal(x0.θ[1], 8°),
        ω=Normal(x0.ω[1], 0.1),
    ))
end


function observation_dist(sce::HovercraftScenario)
    state -> begin
        lmr = landmark_readings(state, sce.landmark_info)
        DistrIterator((landmarks=lmr,))
    end
end

function hover_inputs_transform((; pos, vel, θ, ω), (; ul, ur))
    local loc_v = rotate2d(-θ, vel)
    (; loc_vx=loc_v[1], loc_vy=loc_v[2], ω, θ, ul, ur)
end

function hover_outputs_transform(state, outputs, Δt; clamp_threshold=1e8)
    local (; pos, vel, θ, ω) = state
    local (; loc_ax, loc_ay, der_ω) = outputs
    local acc = rotate2d(θ, @SVector [loc_ax, loc_ay])

    restrict(x) = clamp.(x, -clamp_threshold, clamp_threshold)

    (
        pos=restrict(pos + vel * Δt),
        vel=restrict(vel + acc * Δt),
        θ=restrict(θ + ω * Δt),
        ω=restrict(ω + der_ω * Δt),
    )
end

function hover_outputs_inv_transform(state, state1, Δt)
    local (; θ) = state
    local acc = (state1.vel - state.vel) / Δt
    local loc_acc = rotate2d(-θ, acc)
    local der_ω = (state1.ω - state.ω) / Δt
    (loc_ax=loc_acc[1], loc_ay=loc_acc[2], der_ω=der_ω)
end

function sindy_sketch(sce::HovercraftScenario)
    (; loc_vx, loc_vy, ω, θ, ul, ur, loc_ax, loc_ay, der_ω) = variables(sce)
    input_vars = [loc_vx, loc_vy, ω, θ, ul, ur]
    output_vars = [loc_ax, loc_ay, der_ω]

    MotionModelSketch(
        input_vars,
        output_vars,
        hover_inputs_transform,
        hover_outputs_transform,
        hover_outputs_inv_transform,
    )
end

function batched_sketch(::HovercraftScenario)
    BatchedMotionSketch(;
        state_vars=(; pos=2, vel=2, θ=1, ω=1),
        control_vars=(; ul=1, ur=1),
        input_vars=(; loc_v=2, ω=1, θ=1, ul=1, ur=1),
        output_vars=(; loc_acc=2, a_θ=1),
        state_to_input=state_to_input_SE2,
        output_to_state_rate=output_to_state_rate_SE2,
        output_from_state_rate=output_from_state_rate_SE2,
    )
end

function dynamics_sketch(sce::HovercraftScenario)
    vars = variables(sce)
    inputs = [vars.loc_vx, vars.loc_vy, vars.ω, vars.ul, vars.ur]
    outputs = [vars.loc_ax, vars.loc_ay, vars.der_ω]

    params = OrderedDict{Var,GDistr}(
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
        (; pos, vel, θ, ω, ul, ur, mass, rot_mass, sep, σ_v, σ_ω) = others

        loc_ax = (ul + ur + f_x) / mass
        loc_ay = f_y / mass
        der_ω = ((ur - ul) * sep + f_θ) / rot_mass
        acc = rotate2d(θ, @SVector [loc_ax, loc_ay])
        DistrIterator((
            pos=Dirac(pos + vel * Δt),
            vel=SMvNormal(vel + acc * Δt, σ_v * Δt),
            θ=Dirac(θ + ω * Δt),
            ω=Normal(ω + der_ω * Δt, σ_ω * Δt),
        ))
    end

    DynamicsSketch(; inputs, outputs, params, state_to_inputs, outputs_to_state_dist)
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

function batched_core(::HovercraftScenario, params)
    (input::BatchTuple) -> let
        (; tconf, batch_size) = input
        (; mass, drag_x, drag_y, rot_mass, rot_drag, σ_v, σ_ω) = map(tconf, params)

        (; loc_v, ω, ul, ur) = input.val
        a_x = (ul .+ ur .- drag_x * loc_v[1:1, :]) / mass
        a_y = -drag_y * loc_v[2:2, :] / mass
        a_θ = (ur .- ul .- rot_drag * ω) / rot_mass
        loc_acc = vcat(a_x, a_y)
        @smart_assert size(loc_acc) == size(loc_v)
        μs = BatchTuple(tconf, batch_size, (; loc_acc, a_θ))
        σs = BatchTuple(tconf, batch_size, (loc_acc=tconf([σ_v;;]), a_θ=tconf([σ_ω;;])))
        (; μs, σs)
    end
end

function sindy_core(
    sce::HovercraftScenario, (; σ_v, σ_ω, mass, rot_mass, sep, drag_x, drag_y, rot_drag)
)
    (; loc_vx, loc_vy, ω, θ, ul, ur, loc_ax, loc_ay, der_ω) = map(compile, variables(sce))
    comp_env = ComponentEnv()
    components_scalar_arithmatic!(comp_env)

    comps = (
        loc_ax=LinearExpression(
            0.0, [1 / mass, 1 / mass, -drag_x / mass], [ul, ur, loc_vx], loc_ax.ast.type
        ) |> compile,
        loc_ay=LinearExpression(0.0, [-drag_y / mass], [loc_vy], loc_ay.ast.type) |>
               compile,
        der_ω=LinearExpression(
            0.0, [sep / rot_mass, -sep / rot_mass, -rot_drag], [ur, ul, ω], der_ω.ast.type
        ) |> compile,
    )

    μ_f = combine_functions(comps)
    GaussianGenerator(μ_f, (loc_ax=σ_v, loc_ay=σ_v, der_ω=σ_ω), comps)
end

function plot_scenario!(sce::HovercraftScenario, states, obs_data, name; plt_args...)
    plot_2d_scenario!(states, obs_data, name; sce.landmark_info.landmarks, plt_args...)
end

function plot_trajectories!(::HovercraftScenario, trajectories, name; plt_args...)
    plot_2d_trajectories!(trajectories, name; plt_args...)
end

function get_simplified_motion_model(
    sce::HovercraftScenario, (; σ_v, σ_ω, mass, rot_mass, sep)
)
    sketch = sindy_sketch(sce)
    comps = sindy_core(
        sce, (; σ_v, σ_ω, mass, rot_mass, sep, drag_x=0.0, drag_y=0.0, rot_drag=0.0)
    )
    GaussianMotionModel(sketch, comps)
end

function simulation_controller(sce::HovercraftScenario; noise=0.5)
    pert(x, σ) = x + noise * σ * randn()
    @unzip times, ul_seq, ur_seq = [
        (t=0.0, ul=0.0, ur=0.0),
        (t=0.5, ul=pert(1.0, 0.2), ur=pert(0.4, 0.1)),
        (t=2.0, ul=pert(0.0, 0.2), ur=pert(0.0, 0.1)),
        (t=3.0, ul=pert(0.5, 0.2), ur=pert(0.5, 0.1)),
        (t=5.0, ul=pert(1.1, 0.2), ur=pert(0.5, 0.1)),
        (t=6.0, ul=pert(0.0, 0.2), ur=pert(0.0, 0.1)),
        (t=9.0, ul=pert(0.5, 0.2), ur=pert(1.0, 0.1)),
        (t=10.0, ul=pert(0.0, 0.2), ur=pert(0.4, 0.1)),
        (t=12.0, ul=pert(0.0, 0.2), ur=pert(0.0, 0.1)),
        (t=15.0, ul=0.0, ur=0.0),
    ]
    ul_f = LinearInterpolation(times, ul_seq)
    ur_f = LinearInterpolation(times, ur_seq)
    if rand() < 0.6
        ul_f, ur_f = ur_f, ul_f
    end
    (s, obs, t::Real) -> begin
        (ul=ul_f(t), ur=ur_f(t))
    end
end

function simulation_x0(::HovercraftScenario)
    (
        pos=@SVector([0.5 + 2randn(), 0.5 + 2randn()]),
        vel=@SVector([0.25 + 0.3randn(), 0.0 + 0.2randn()]),
        θ=randn(),
        ω=10° * randn(),
    )
end

function simulation_params(::HovercraftScenario)
    (;
        mass=1.5,
        drag_x=0.06,
        drag_y=0.10,
        rot_mass=1.5,
        rot_drag=0.07,
        sep=0.81,
        σ_v=0.04,
        σ_ω=0.03,
    )
end

function hovercraft_scenario()
    landmarks = @SVector([
        @SVector([-1.0, 2.5]),
        @SVector([1.0, -1.0]),
        @SVector([8.0, -5.5]),
        @SVector([14.0, 6.0]),
        @SVector([16.0, -7.5])
    ])
    HovercraftScenario(LandmarkInfo(; landmarks))
end

state_L2_loss(::HovercraftScenario) = L2_in_SE2