using StatsFuns: softmax!, logsumexp, softmax
using StatsBase: countmap

export simulate_trajectory, states_log_score, data_likelihood

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
    (; x0_dist, motion_model), (;times, controls), states, ::Type{T}
)::T where T
    p::T = log_score(x0_dist, states[1], T)
    for i in 1:length(states)-1
        Δt = times[i+1]-times[i]
        state_distr = motion_model(states[i], controls[i], Δt)
        p += log_score(state_distr, states[i+1], T)
    end
    p
end

function data_likelihood((; obs_model), (; observations), states)
    observations::Dict{Int, <:Any}
    foldl(observations; init=0.0) do s, (t, obs)
        s + logpdf(obs_model(states[t], obs))
    end
end

struct MarkovSystem{X, X0_Dist, Motion, ObsM}
    x0_dist::X0_Dist
    "motion_model(x, control, Δt) -> distribution_of_x′"
    motion_model::Motion
    "obs_model(x) -> distribtion_of_y"
    obs_model::ObsM
end

MarkovSystem(x0_dist::A, motion_model::B, obs_model::C) where {A,B,C} = begin
    X = typeof(rand(x0_dist))
    MarkovSystem{X, A, B, C}(x0_dist, motion_model, obs_model)
end

Base.show(io::IO, sys::MarkovSystem) = begin
    print(io, "MarkovSystem", (; sys.x0_dist, sys.motion_model, sys.obs_model))
end

effective_particles(weights::AbstractVector) = 1/sum(abs2, weights)

function systematic_resample!(indices, weights, bins_buffer)
    N = length(weights)
    M = length(indices)
    @assert length(bins_buffer) == N
    bins_buffer[1] = weights[1]
    for i = 2:N
        bins_buffer[i] = bins_buffer[i-1] + weights[i]
    end
    r = rand()*bins_buffer[end]/M
    s = r:(1/M):(bins_buffer[end]+r) # Added r in the end to ensure correct length (r < 1/N)
    bo = 1
    for i = 1:M
        @inbounds for b = bo:N
            if s[i] < bins_buffer[b]
                indices[i] = b
                bo = b
                break
            end
        end
    end
    Random.shuffle!(indices)
    return indices
end

function systematic_resample(weights::AbstractArray{N}, n_samples::Integer) where N
    indices = collect(1:n_samples)
    bins_buffer = fill(zero(N), length(weights))
    systematic_resample!(indices, weights, bins_buffer)
end

"""
Run particle filters forward in time
"""
function forward_filter(
    system::MarkovSystem{X}, (; times, obs_frames, controls, observations), n_particles; 
    resample_threshold::Float64 = 0.5, score_f=logpdf,   
) where X
    @assert eltype(obs_frames) <: Integer
    T, N = length(times), n_particles
    (; x0_dist, motion_model, obs_model) = system
    
    particles = Matrix{X}(undef, N, T)
    ancestors = Matrix{Int}(undef, N, T)
    log_weights = Matrix{Float64}(undef, N, T) # unnormalized log weights
    weights = Matrix{Float64}(undef, N, T) # normalized probabilities

    particles[:, 1] .= (rand(x0_dist) for _ in 1:N)
    ancestors[:, :] .= 1:N
    
    current_lw = fill(-log(N), N) # log weights at the current time step
    current_w = exp.(current_lw)
    log_obs::Float64 = 0.0  # estimation of the log observation likelihood
    bins_buffer = zeros(Float64, N) # used for systematic resampling

    progress = Progress(T-1, desc="forward_filter", output=stdout)
    for t in 1:T
        if t > 1 
            # optionally resample
            if effective_particles(current_w) < N * resample_threshold
                indices = @views(ancestors[:, t-1])
                systematic_resample!(indices, current_w, bins_buffer)
                current_lw .= -log(N)
                particles[:, t-1] = particles[indices, t-1]
            end
            Δt = times[t] - times[t-1]
            u = controls[t-1]
            Threads.@threads for i in 1:N
                particles[i, t] = rand(motion_model(particles[i, t-1], u, Δt))
            end
        end

        if t in obs_frames
            Threads.@threads for i in 1:N
                current_lw[i] += score_f(obs_model(particles[i, t]), observations[t])
            end
        end

        log_z_t = logsumexp(current_lw)
        current_lw .-= log_z_t
        current_w .= exp.(current_lw)
        log_weights[:, t] = current_lw
        weights[:, t] = current_w
        log_obs += log_z_t
        next!(progress)
    end

    log_obs::Float64
    (; particles, weights, log_weights, ancestors, log_obs)
end


"""
Sample smooting trajecotries by tracking the ancestors of a particle filter.
"""
function filter_smoother(system, obs_data; 
    n_particles, n_trajs,
    resample_threshold=0.5, score_f=logpdf,
)
    (; particles, log_weights, ancestors, log_obs) = forward_filter(
        system, obs_data, n_particles; resample_threshold, score_f)
    traj_ids = systematic_resample(softmax(log_weights[:, end]), n_trajs)
    trajectories = particle_trajectories(particles, ancestors, traj_ids)
    (; trajectories, log_obs)
end

"""
From the give particle `indices` at the last time step, going backward to trace
out the ancestral lineages. 
"""
function particle_trajectories(
    particles::Matrix{P}, ancestors::Matrix{Int}, indices::Vector{Int},
)::Matrix{P} where P
    N, T = size(particles)
    indices = copy(indices)
    M = length(indices)
    trajs = Matrix{P}(undef, M, T)
    trajs[:, T] = particles[indices, T]
    for t in T-1:-1:1
        indices .= ancestors[indices, t+1]
        trajs[:, t] = particles[indices, t]
    end
    trajs
end

"""
Particle smoother based on the Forward filtering-backward sampling algorithm.
"""
function ffbs_smoother(
    system::MarkovSystem{X}, obs_data; 
    n_particles, n_trajs,
    resample_threshold::Float64 = 0.5,
    score_f=logpdf,
    progress_offset=0,
) where X
    (; times, obs_frames, controls, observations) = obs_data

    function forward_logp(x_t, x_t′, t)
        local Δt = times[t+1]-times[t]
        local d = system.motion_model(x_t, controls[t], Δt)
        score_f(d, x_t′)
    end

    (; particles, log_weights, log_obs) = forward_filter(
        system, obs_data, n_particles; resample_threshold, score_f)
    log_obs::Float64
    trajectories = backward_sample(forward_logp, particles, log_weights, n_trajs; progress_offset)
    (; trajectories, log_obs)
end


"""
Performs the backward recursion of the Forward filtering-backward sampling algorithm
to sample from the smoothing distribution.

## Arguments
- `forward_logp(x_t, x_t′, t)` should return the log probability of the transition dynamics.
"""
function backward_sample(
    forward_logp::Function,
    filter_particles::Matrix{P}, 
    filter_log_weights::Matrix{<:Real}, 
    n_trajs::Integer;
    progress_offset=0,
) where P
    M = n_trajs
    N, T = size(filter_particles)
    trajectories = Matrix{P}(undef, M, T)
    weights = [softmax(filter_log_weights[:, T]) for _ in 1:M]

    trajectories[:, T] .= (
        let j = rand(Categorical(weights[j]))
            filter_particles[j, T]
        end for j in 1:M)

    progress = Progress(T-1, desc="backward_sample", offset=progress_offset, output=stdout)
    for t in T-1:-1:1
        Threads.@threads for j in 1:M
            weights[j] .= @views(filter_log_weights[:, t]) .+
                forward_logp.(@views(filter_particles[:, t]), Ref(trajectories[j, t+1]), t)
            softmax!(weights[j])
            j′ = rand(Categorical(weights[j]))
            trajectories[j, t] = filter_particles[j′, t]
        end
        next!(progress)
    end
    trajectories
end

function log_softmax(x::AbstractArray{<:Real})
    u = maximum(x)
    x = x .- u
    dnum = logsumexp(x)
    x .- dnum
end