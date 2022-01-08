struct RealCarScenario <: Scenario end

Base.summary(io::IO, ::RealCarScenario) = print(io, "RealCarScenario")

function batched_sketch(::RealCarScenario)
    state_vars = (; pos=2, angle_2d=2, vel=2, ω_2d=2)
    control_vars = (; twist_linear=3, twist_angular=3)
    input_vars = (; angle_2d=2, vel=2, ω_2d=2, control_vars...)
    output_vars = (; a_linear=2, a_rot=2)
    state_to_input = (x, u) -> BatchTuple(x, u) do (; angle_2d, vel, ω_2d), uval
        (; angle_2d, vel, ω_2d, uval...)
    end

    output_to_state_rate = (x, o) -> BatchTuple(x, o) do (; vel, ω_2d), (; a_linear, a_rot)
        (pos=vel, angle_2d=ω_2d, vel=a_linear, ω_2d=a_rot)
    end

    output_from_state_rate = (x, dxdt) -> BatchTuple(dxdt) do (; vel, ω_2d)
        (a_linear=vel, a_rot=ω_2d)
    end

    BatchedMotionSketch(;
        state_vars,
        control_vars,
        input_vars,
        output_vars,
        state_to_input,
        output_to_state_rate,
        output_from_state_rate,
    )
end

