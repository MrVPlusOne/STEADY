abstract type PosteriorSampler end


"""
    sample_posterior(::PosteriorSampler, system, obs_data) -> (; trajectories, log_obs)
where `trajectories::Vector{Vector{State}}` and `log_obs::Vector{Float64}`.
"""
function sample_posterior end
# function initialize!, finish! end

struct ParticleFilterSampler{n_threads} <: PosteriorSampler
    n_particles::Int
    n_trajs::Int
    n_runs::Int
    caches::Vector{Dict{Symbol, Any}}
end

ParticleFilterSampler(; n_particles, n_trajs, n_runs, n_threads=n_runs) = 
    ParticleFilterSampler{n_threads}(
        n_particles, n_trajs, n_runs, [Dict{Symbol, Any}() for _ in 1:n_threads])

function Base.empty!(pf::ParticleFilterSampler)
    foreach(empty!, pf.caches)
end

function sample_posterior(
    pf::ParticleFilterSampler{n_threads}, system, obs_data,
) where n_threads
    (; n_particles, n_trajs, n_runs) = pf
    results = parallel_map(1:n_runs, nothing; n_threads) do _, i
        tid = Threads.threadid()-1
        @assert 1 <= tid <= n_threads
        filter_smoother(system, obs_data, use_auxiliary_proposal=false;
            n_particles, n_trajs, cache=pf.caches[tid], showprogress=i==1)
    end
    @unzip_named (log_obs, :log_obs), (trajectories_list, :trajectories) = results
    trajectories = reduce(vcat, trajectories_list)
    (;trajectories, log_obs)
end
