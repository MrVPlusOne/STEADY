"""
The 2D hovercraft scenario.
"""
struct HovercraftScenario <: Scenario end

Base.summary(io::IO, ::HovercraftScenario) = print(io, "HovercraftScenario")

function batched_sketch(::HovercraftScenario)
    control_vars = (; ul=1, ur=1)  # thrust left and thrust right
    batched_sketch_SE2(control_vars)
end

function state_L2_loss_batched(::HovercraftScenario)
    L2_in_SE2_batched
end

function pose_to_opt_vars(::HovercraftScenario)
    pose_to_opt_vars_SE2
end

function pose_from_opt_vars(::HovercraftScenario)
    pose_from_opt_vars_SE2
end

function get_simplified_motion_core(::HovercraftScenario)
    mass = 1.5f0
    rot_mass = 1.5f0
    acc_σs = (a_loc=0.2f0, a_rot=0.1f0)
    (core_input::BatchTuple, Δt) -> begin
        local μs = BatchTuple(core_input) do (; ul, ur)
            a_x = (ul .+ ur) ./ mass
            a_y = zero(a_x)
            a_loc = vcat(a_x, a_y)

            a_rot = (ur .- ul) ./ rot_mass
            (; a_loc, a_rot)
        end
        local σs = BatchTuple(core_input) do _
            acc_σs
        end
        (; μs, σs)
    end
end
##----------------------------------------------------------- 
# simulation utilities
function batched_core(::HovercraftScenario, params)
    (input::BatchTuple, Δt) -> let
        (; tconf, batch_size) = input
        (; mass, drag_x, drag_y, rot_mass, rot_drag, σ_v, σ_ω) = map(tconf, params)

        (; loc_v, ω, ul, ur) = input.val
        a_x = (ul .+ ur .- drag_x * loc_v[1:1, :]) / mass
        a_y = -drag_y * loc_v[2:2, :] / mass
        a_rot = (ur .- ul .- rot_drag * ω) / rot_mass
        a_loc = vcat(a_x, a_y)
        @smart_assert size(a_loc) == size(loc_v)
        μs = BatchTuple(tconf, batch_size, (; a_loc, a_rot))
        σs = BatchTuple(input, (a_loc=tconf(fill(σ_v, 2, 1)), a_rot=tconf(fill(σ_ω, 1, 1))))
        (; μs, σs)
    end
end

function simulation_params(::HovercraftScenario)
    (;
        mass=1.5,
        drag_x=0.06,
        drag_y=0.10,
        rot_mass=1.5,
        rot_drag=0.07,
        sep=0.81,
        σ_v=0.08,
        σ_ω=0.05,
    )
end

function simulation_controller(::HovercraftScenario; noise=0.5)
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
    θ = π / 5 + π * randn()
    (;
        pos=[0.5, 1.0] + 2randn(2),
        angle_2d=[cos(θ), sin(θ)],
        vel=[0.25 + 0.3randn(), 0.1 + 0.2randn()],
        ω=[π / 50 + 0.1π * randn()],
    )
end
##-----------------------------------------------------------