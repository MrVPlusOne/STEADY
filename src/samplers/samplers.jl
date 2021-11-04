using StatsFuns: softmax!, logsumexp, softmax
using StatsBase: countmap

export MarkovSystem
struct MarkovSystem{X, X0_Dist, Motion, ObsM}
    x0_dist::X0_Dist
    "motion_model(x, control, Δt) -> distribution_of_x′"
    motion_model::Motion
    "obs_model(x) -> distribtion_of_y"
    obs_model::ObsM
end

MarkovSystem(x0_dist::A, motion_model::B, obs_model::C) where {A<:GDistr,B,C} = begin
    X = typeof(rand(x0_dist))
    MarkovSystem{X, A, B, C}(x0_dist, motion_model, obs_model)
end

Base.show(io::IO, sys::MarkovSystem) = begin
    print(io, "MarkovSystem", (; sys.x0_dist, sys.motion_model, sys.obs_model))
end

export simulate_trajectory, states_log_score, data_log_score

function simulate_trajectory(times, x0, (; motion_model, obs_model), controller)
    T = length(times)
    state = x0

    states = []
    observations = []
    controls = []

    for t in 1:T
        y = rand(obs_model(state))
        u = controller(state, y, times[t])

        push!(states, state)
        push!(observations, y)
        push!(controls, u)

        if t < T
            Δt = times[t+1] - times[t]
            state = rand(motion_model(state, u, Δt))
        end
    end

    (states=specific_elems(states), 
        observations=specific_elems(observations), 
        controls=specific_elems(controls))
end

function states_log_score(
    motion_model::Function, (;times, controls, x0_dist), states, ::Type{T},
)::T where T
    p::T = log_score(x0_dist, states[1], T)
    for i in 1:length(states)-1
        Δt = times[i+1]-times[i]
        state_distr = motion_model(states[i], controls[i], Δt)
        p += log_score(state_distr, states[i+1], T)
    end
    p
end

function data_log_score(
    obs_model::Function, (; times, obs_frames, observations), states, ::Type{T}
)::T where T
    p::T = 0.0
    for t in obs_frames
        t::Integer
        p += log_score(obs_model(states[t]), observations[t], T)
    end
    p
end

function data_logp(
    (; obs_model), (; times, obs_frames, observations), states,
)
    sum(logpdf(obs_model(states[t]), observations[t]) for t in obs_frames)
end

function total_log_score(system, obs_data, states, ::Type{T}) where T
    states_log_score(system.motion_model, obs_data, states, T) + 
        data_log_score(system.obs_model, obs_data, states, T)
end

function log_softmax(x::AbstractArray{<:Real})
    u = maximum(x)
    x = x .- u
    dnum = logsumexp(x)
    x .- dnum
end

export map_trajectory
function map_trajectory(
    system::MarkovSystem{X},
    obs_data,
    traj_guess::Vector{X};
    optim_options::Optim.Options=Optim.Options(f_abstol=1e-4),
) where X
    (; times, obs_frames, controls, observations) = obs_data
    
    function loss(vec::AbstractVector{T})::T where T
        local traj = structure_from_vec(traj_guess, vec)
        local total_score = states_log_score(system.motion_model, obs_data, traj, T) + 
            data_log_score(system.obs_model, obs_data, traj, T)
        -total_score
    end

    traj_guess_vec = structure_to_vec(traj_guess)
    f_init = -loss(traj_guess_vec)
    sol = optimize_no_tag(loss, traj_guess_vec, optim_options)

    traj_final_vec = Optim.minimizer(sol)
    f_final = -loss(traj_final_vec)
    traj_final = structure_from_vec(traj_guess, traj_final_vec)
    stats = (converged=Optim.converged(sol), iterations=Optim.iterations(sol))
    (; states=traj_final, f_init, f_final, stats)
end

include("particle_filters.jl")
include("MCMC_samplers.jl")