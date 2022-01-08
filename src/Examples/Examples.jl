export generate_data
function generate_data(
    x₀::NamedTuple,
    x′₀::NamedTuple,
    f_x′′::Function,
    params::NamedTuple,
    others::NamedTuple,
    times::TimeSeries;
    observe, # observe(s, s_prev, others; noise_scale) -> obs
    controller, # controller(s, obs) -> action
    noise_scale,
    allow_state_access=false, # whether to also pass the current state to the controller
)
    # Refs to store the current time step, state, and observation
    i_ref = Ref(1)
    s_ref = Ref(merge(x₀, x′₀))
    obs_ref = Ref(observe(s_ref[], s_ref[], others; noise_scale))

    states = NamedTuple[]
    observations = NamedTuple[]
    actions = NamedTuple[]

    should_stop() = i_ref[] > length(times)

    next_time_action!() = begin
        i = i_ref[]
        i_ref[] += 1
        act =
            allow_state_access ? controller(s_ref[], obs_ref[]) : controller(obs_ref[])
        push!(actions, act)
        times[i], act
    end

    record_state!(s) = begin
        s_prev = s_ref[]
        s_ref[] = s
        obs_ref[] = observe(s, s_prev, others; noise_scale)
        push!(states, s_ref[])
        push!(observations, obs_ref[])
    end

    SEDL.simulate(x₀, x′₀, f_x′′, params, should_stop, next_time_action!, record_state!)
    (; params, others, states, observations, actions, times)
end

include("scenario_utils.jl")
include("RealCar2D.jl")
include("Car1D.jl")
include("Car2D.jl")
include("Hovercraft2D.jl")
include("HovercraftBatched.jl")
include("Rocket2D.jl")
include("simulation_experiments.jl")