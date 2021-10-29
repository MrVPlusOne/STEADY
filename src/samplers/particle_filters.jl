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
        @inbounds for b in bo:N
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
    use_auxiliary_proposal::Bool=true, showprogress=true,
) where X
    @assert eltype(obs_frames) <: Integer
    T, N = length(times), n_particles
    (; x0_dist, motion_model, obs_model) = system
    
    particles = Matrix{X}(undef, N, T)
    ancestors = Matrix{Int}(undef, N, T)

    particles[:, 1] .= (rand(x0_dist) for _ in 1:N)
    ancestors[:, :] .= 1:N
    
    log_weights = fill(-log(N), N) # log weights at the current time step
    weights = exp.(log_weights)
    log_obs::Float64 = 0.0  # estimation of the log observation likelihood
    bins_buffer = zeros(Float64, N) # used for systematic resampling
    if use_auxiliary_proposal
        aux_weights = zeros(Float64, N)
        aux_log_weights = zeros(Float64, N)
        aux_indices = collect(1:N)
    end

    progress = Progress(T-1, desc="forward_filter", output=stdout, enabled=showprogress)
    for t in 1:T
        if t in obs_frames
            for i in 1:N
                log_weights[i] += score_f(obs_model(particles[i, t]), observations[t])
            end
            log_z_t = logsumexp(log_weights)
            log_weights .-= log_z_t
            weights .= exp.(log_weights)
            log_obs += log_z_t

            # optionally resample
            if effective_particles(weights) < N * resample_threshold
                indices = @views(ancestors[:, t])
                systematic_resample!(indices, weights, bins_buffer)
                log_weights .= -log(N)
                particles[:, t] = particles[indices, t]
            end
        end

        if t < T
            Δt = times[t+1] - times[t]
            u = controls[t]
            if use_auxiliary_proposal && (t+1) in obs_frames
                # use the mean of the dynamics to predict observation likelihood and 
                # resample particles accordingly
                @inbounds for i in 1:N
                    μ = mean(motion_model(particles[i, t], u, Δt))
                    aux_log_weights[i] = score_f(obs_model(μ), observations[t+1])
                end
                aux_log_weights .-= logsumexp(aux_log_weights)
                aux_weights .= exp.(aux_log_weights)
                aux_indices .= 1:N
                systematic_resample!(aux_indices, aux_weights, bins_buffer)
                ancestors[:, t] = ancestors[aux_indices, t]
                particles[:, t] = particles[aux_indices, t]
                log_weights -= aux_log_weights[aux_indices]
            end

            @inbounds for i in 1:N
                particles[i, t+1] = rand(motion_model(particles[i, t], u, Δt))
            end
        end
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
    use_auxiliary_proposal::Bool=false,
    showprogress=true,
)
    (; particles, log_weights, ancestors, log_obs) = forward_filter(
        system, obs_data, n_particles; resample_threshold, score_f, 
        use_auxiliary_proposal, showprogress)
    traj_ids = systematic_resample(softmax(log_weights), n_trajs)
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