using Base.Threads: @threads

"""
Particle Gibbs with Ancestor Sampling MCMC Kernel.
"""
function PGAS_kernel(
    system::MarkovSystem{X}, (; times, obs_frames, controls, observations); 
    n_particles, resample_threshold::Float64, showprogress=true,
) where X
    @smart_assert eltype(obs_frames) <: Integer
    T, N = length(times), n_particles
    (; x0_dist, motion_model, obs_model) = system

    particles = Matrix{X}(undef, N, T)
    ancestors = Matrix{Int}(undef, N, T)
    log_weights = Vector{Float64}(undef, N) # unnormalized log weights
    weights = Vector{Float64}(undef, N) # normalized probabilities
    transition_log_weights = Vector{Float64}(undef, N) # normalized probabilities
    bins_buffer = zeros(Float64, N)

    ref_traj::Vector{X} -> begin
        particles[1:N-1, 1] .= (rand(x0_dist) for _ in 1:N-1)
        particles[N, :] = ref_traj
        
        ancestors[:, :] .= 1:N
        weights .= 1 / N
        log_weights .= -log(N) 

        showprogress && (progress = Progress(T-1, desc="PG Kernel", output=stdout))
        for t in 1:T
            if t > 1
                Δt = times[t] - times[t-1]
                u = controls[t-1]
                
                # for particle N, perform ancestor resampling
                transition_log_weights .= log_weights
                for i in 1:N
                    transition_log_weights[i] += 
                        logpdf(motion_model(particles[i, t-1], u, Δt), ref_traj[t])
                end
                j = rand(Categorical(softmax!(transition_log_weights)))
                ancestors[N, t-1] = j
                past = particles[j, t-1]
                
                if effective_particles(weights) < N * resample_threshold
                    # for particle 1..N-1, perform normal resampling 
                    indices = @views(ancestors[1:N-1, t-1])
                    systematic_resample!(indices, weights, bins_buffer)
                    log_weights .= -log(N) 
                    particles[1:N-1, t-1] = particles[indices, t-1]
                end
                particles[N, t-1] = past
                # forward simulation
                for i in 1:N-1
                    particles[i, t] = rand(motion_model(particles[i, t-1], u, Δt))
                end
            end

            if t in obs_frames
                for i in 1:N
                    log_weights[i] += logpdf(obs_model(particles[i, t]), observations[t])
                end
            end

            log_z_t = logsumexp(log_weights)            
            log_weights .-= log_z_t
            weights .= exp.(log_weights)

            showprogress && next!(progress)
        end

        k = rand(Categorical(weights))
        new_traj = Vector{X}(undef, T)
        new_traj[T] = particles[k, T]
        for t in T-1:-1:1
            k = ancestors[k, t+1]
            new_traj[t] = particles[k, t]
        end
        new_traj
    end
end

function PGAS_smoother(
    system::MarkovSystem{X}, obs_data, init_traj; 
    n_particles, n_trajs, n_thining, n_burn_in=n_trajs, resample_threshold=0.6,
    showprogress=true,
) where X
    kernel = PGAS_kernel(system, obs_data; n_particles, resample_threshold)
    trajectories = Vector{X}[]
    state = init_traj
    progress = Progress(n_burn_in+n_trajs, desc="PGAS_smoother", enabled=showprogress, output=stdout)
    for i in 1:n_burn_in+n_trajs
        for _ in 1:n_thining
            state = kernel(state)
        end
        next!(progress)
        i > n_burn_in && (push!(trajectories, state))
    end
    (; trajectories, state, log_obs=0.0) # doesn't support log_obs estimation
end