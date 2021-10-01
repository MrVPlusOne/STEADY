using StatsFuns: softmax!

struct StochasticSystem{X, U, Y, X0_Dist, Motion, ObsM}
    x0_dist::X0_Dist
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
    return indices
end

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