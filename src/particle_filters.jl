using StatsFuns: softmax!
using StatsBase: countmap

struct StochasticSystem{X, U, Y, X0_Dist, Motion, ObsM}
    x0_dist::X0_Dist
    "motion_model(x, control, Δt) -> distribution_of_x′"
    motion_model::Motion
    obs_model::ObsM
end

StochasticSystem(x0_dist::A, motion_model::B, obs_model::C; X, U, Y) where {A,B,C} = 
    StochasticSystem{X, U, Y, A, B, C}(x0_dist, motion_model, obs_model)

Base.show(io::IO, sys::StochasticSystem) = begin
    print(io, "StochasticSystem", (; sys.x0_dist, sys.motion_model, sys.obs_model))
end

effective_particles(weights::AbstractVector) = 1/sum(abs2, weights)

function resample!(indices, weights, bins_buffer)
    N = length(weights)
    bins_buffer[1] = weights[1]
    for i = 2:N
        bins_buffer[i] = bins_buffer[i-1] + weights[i]
    end
    r = rand()*bins_buffer[end]/N
    s = r:(1/N):(bins_buffer[N]+r) # Added r in the end to ensure correct length (r < 1/N)
    bo = 1
    for i = 1:N
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

"""
Run particle filters forward in time
"""
function forward_filter(
    system::StochasticSystem{X}, (; times, controls, observations), n_particles; 
    resample_threshold::Float64 = 0.5,    
) where X
    T, N = length(times), n_particles
    (; x0_dist, motion_model, obs_model) = system
    
    particles = Matrix{X}(undef, N, T)
    ancestors = Matrix{Int}(undef, N, T)
    log_weights = Matrix{Float64}(undef, N, T) # unnormalized log weights
    weights = Matrix{Float64}(undef, N, T) # normalized probabilities

    particles[:, 1] .= (rand(x0_dist) for _ in 1:N)
    ancestors[:, :] .= 1:N
    
    # log weights at the current time step
    current_lw = logpdf.(obs_model.(@views(particles[:, 1])), Ref(observations[1]))
    bins_buffer = zeros(Float64, N)

    for t in 1:T-1
        w_vec = @views weights[:, t]
        softmax!(w_vec, current_lw)
        log_weights[:, t] = current_lw
        
        # optionally resample
        if effective_particles(w_vec) < N * resample_threshold
            indices = @views(ancestors[:, t])
            resample!(indices, w_vec, bins_buffer)
            current_lw .= 0.0
            particles[:, t] = particles[indices, t]
        end

        Δt = times[t+1] - times[t]
        u = controls[t]
        particles[:, t+1] .= rand.(motion_model.(@views(particles[:, t]), Ref(u), Δt))
        current_lw .+= logpdf.(obs_model.(@views(particles[:, t+1])), Ref(observations[t+1]))
    end
    log_weights[:, T] = current_lw
    softmax!(@views(weights[:, T]), current_lw)

    (; particles, weights, log_weights, ancestors)
end

"""
Similar to particle filters, but try to track the data likelihood of the entire trajectory 
by not resetting weights to 0 during resampling.
"""
function forward_filter_tracked(
    system::StochasticSystem{X}, (; times, controls, observations), n_particles; 
    resample_threshold::Float64 = 0.5,    
) where X
    T, N = length(times), n_particles
    (; x0_dist, motion_model, obs_model) = system
    
    particles = Matrix{X}(undef, N, T)
    ancestors = Matrix{Int}(undef, N, T)
    log_weights = Matrix{Float64}(undef, N, T) # unnormalized log weights
    weights = Matrix{Float64}(undef, N, T) # normalized probabilities

    particles[:, 1] .= (rand(x0_dist) for _ in 1:N)
    ancestors[:, :] .= 1:N
    
    # log weights at the current time step
    current_lw = logpdf.(obs_model.(@views(particles[:, 1])), Ref(observations[1]))
    bins_buffer = zeros(Float64, N)

    for t in 1:T-1
        w_vec = @views weights[:, t]
        softmax!(w_vec, current_lw)
        log_weights[:, t] = current_lw
        
        # optionally resample
        if effective_particles(w_vec) < N * resample_threshold
            indices = @views(ancestors[:, t])
            resample!(indices, w_vec, bins_buffer)
            index_to_log_n = Dict(i => -log(n) for (i, n) in countmap(indices))
            for i in 1:N
                ancestor = indices[i]
                current_lw[i] = log_weights[ancestor, t] + index_to_log_n[ancestor]
            end
            particles[:, t] = particles[indices, t]
        end

        Δt = times[t+1] - times[t]
        u = controls[t]
        particles[:, t+1] .= rand.(motion_model.(@views(particles[:, t]), Ref(u), Δt))
        current_lw .+= logpdf.(obs_model.(@views(particles[:, t+1])), Ref(observations[t+1]))
    end
    log_weights[:, T] = current_lw
    softmax!(@views(weights[:, T]), current_lw)

    (; particles, weights, log_weights, ancestors)
end

"""
Sample smooting trajecotries by tracking the ancestors of a particle filter.
"""
function filter_smoother(system, data, n_particles; 
    resample_threshold=1.0, thining=10, track_weights=false
)
    pf = track_weights ? forward_filter_tracked : forward_filter
    (; particles, log_weights, ancestors) = pf(system, data, n_particles; resample_threshold)
    trajs = particle_trajectories(particles, ancestors)
    trajectories = trajs[1:thining:end,:]
    log_weights = log_weights[1:thining:end, end]
    log_weights .-= logsumexp(log_weights)
    (; trajectories, log_weights)
end

"""
Particle smoother based on the Forward filtering-backward sampling algorithm.
"""
function ffbs_smoother(
    system::StochasticSystem{X}, (; times, controls, observations); 
    n_particles, n_trajs,
    resample_threshold::Float64 = 0.5,
    progress_offset=0,
) where X
    function forward_logp(x_t, x_t′, t)
        local Δt = times[t+1]-times[t]
        local d = system.motion_model(x_t, controls[t], Δt)
        logpdf(d, x_t′)
    end

    (particles, log_weights) = forward_filter(
        system, (; times, controls, observations), n_particles; resample_threshold)
    backward_sample(forward_logp, particles, log_weights, n_trajs; progress_offset)
end


"""
Performs the backward recursion of the Forward filtering-backward sampling algorithm
to sample from the smoothing distribution.

## Arguments
- `forward_logp(x_t, x_t′, t)` should return the log likelihood of the transition dynamics.
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
    particles = Matrix{P}(undef, M, T)
    weights = [softmax(filter_log_weights[:, T]) for _ in 1:M]

    particles[:, T] .= (
        let j = rand(Categorical(weights[j]))
            filter_particles[j, T]
        end for j in 1:M)

    progress = Progress(T-1, desc="backward_sample", offset=progress_offset, output=stdout)
    for t in T-1:-1:1
        Threads.@threads for j in 1:M
            # weights[j] .= @views(filter_log_weights[:, t]) .+
            #     forward_logp.(@views(filter_particles[:, t]), Ref(particles[j, t+1]), t)
            for k in 1:N
                weights[j][k] = filter_log_weights[k, t] + 
                    forward_logp(filter_particles[k, t], particles[j, t+1], t)
            end
            softmax!(weights[j])
            j′ = rand(Categorical(weights[j]))
            particles[j, t] = filter_particles[j′, t]
        end
        next!(progress)
    end
    log_weights = fill(log(1/M), M)
    particles, log_weights
end

function log_softmax(x::AbstractArray{<:Real})
    u = maximum(x)
    x = x .- u
    dnum = logsumexp(x)
    x .- dnum
end