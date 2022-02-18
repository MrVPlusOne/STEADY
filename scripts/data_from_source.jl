using Base: @kwdef

abstract type DataSource end

@kwdef struct SimulationData <: DataSource
    n_train_ex::Int
    n_valid_ex::Int
    n_test_ex::Int
    times
    Δt::Real = times[2] - times[1]
end

@kwdef struct SeparateData <: DataSource
    train_data_dir::String
    valid_data_dir::String
    test_data_dir::String
end

@kwdef struct MixedData <: DataSource
    data_dir
    test_data_ratio::Real
    valid_data_ratio::Real
end

function data_from_source(
    sce::SEDL.Scenario, src::SimulationData, tconf::TensorConfig; motion_model, obs_model
)
    (; n_train_ex, n_valid_ex, n_test_ex, times, Δt) = src

    @smart_assert isempty(Flux.params(motion_model))

    function mk_batch_controller(n_exs)
        let controllers = [SEDL.simulation_controller(sce) for _ in 1:n_exs]
            (args...) -> begin
                us = [map(v -> [v], ctrl(args...)) for ctrl in controllers]
                SEDL.BatchTuple(tconf, us)
            end
        end
    end

    sample_next_state = (x -> x.next_state) ∘ motion_model
    sample_observation = rand ∘ obs_model

    function mk_sim_data(n_exs)
        local controller = mk_batch_controller(n_exs)
        local x0_batch = SEDL.BatchTuple(tconf, [SEDL.simulation_x0(sce) for _ in 1:n_exs])
        local data = SEDL.simulate_trajectory(
            times, x0_batch, sample_next_state, sample_observation, controller
        )
        merge(data, (; Δt=tconf(Δt)))
    end

    (
        test=mk_sim_data(n_test_ex),
        valid=mk_sim_data(n_valid_ex),
        train=mk_sim_data(n_train_ex),
    )
end

function data_from_source(sce::SEDL.Scenario, src::SeparateData, tconf::TensorConfig; obs_model)
    (; train_data_dir, valid_data_dir, test_data_dir) = src

    map((test=test_data_dir, valid=valid_data_dir, train=train_data_dir)) do path
        data = SEDL.Dataset.read_data_from_csv(sce, path, tconf)
        observations = (x -> rand(obs_model(x))).(data.states)
        Δt = tconf(data.times[2] - data.times[1])
        merge(data, (; observations, Δt))
    end
end

function data_from_source(
    sce::SEDL.Scenario, src::MixedData, tconf::TensorConfig; obs_model, shuffle_seed=123
)
    (; data_dir, test_data_ratio, valid_data_ratio) = src
    @smart_assert 0 < test_data_ratio + valid_data_ratio < 1

    rng = Random.MersenneTwister(shuffle_seed)

    (; n_trajs, times, states, controls) = SEDL.Dataset.read_data_from_csv(
        sce, data_dir, tconf
    )
    ids = Random.shuffle(rng, 1:n_trajs)
    n_test = ceil(Int, n_trajs * test_data_ratio)
    n_valid = ceil(Int, n_trajs * valid_data_ratio)

    observations = [rand(obs_model(x)) for x in states]
    Δt = tconf(times[2] - times[1])

    ids_tuple = (
        test=ids[1:n_test],
        valid=ids[(n_test + 1):(n_test + n_valid)],
        train=ids[(n_test + n_valid + 1):end],
    )

    map(ids_tuple) do ids
        (;
            times,
            states=[s[ids] for s in states],
            controls=[c[ids] for c in controls],
            observations=[o[ids] for o in observations],
            Δt,
        )
    end
end

function generate_or_load(gen_fn, path, result_name; should_warn=true)
    if isfile(path)
        should_warn && @warn "Loading $result_name from $path..."
        deserialize(path)
    else
        result = gen_fn()
        should_warn && @warn "Saving $result_name to $path..."
        mkpath(dirname(path))
        serialize(path, result)
        result
    end
end