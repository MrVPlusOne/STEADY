abstract type DataSource end
@kwdef struct SimulationData <: DataSource
    n_train_ex::Int
    n_test_ex::Int
    times
    Δt::Real = times[2] - times[1]
end
@kwdef struct RealData <: DataSource
    train_data_path::String
    test_data_path::String
end

function data_from_source(src::SimulationData, tconf::TensorConfig; motion_model, obs_model)
    (; n_train_ex, n_test_ex, times, Δt) = src

    sample_x0() = (;
        pos=[0.5, 1.0] + 2randn(2),
        vel=[0.25 + 0.3randn(), 0.1 + 0.2randn()],
        θ=[π / 5 + π * randn()],
        ω=[π / 50 + 0.1π * randn()],
    )

    @smart_assert isempty(Flux.params(motion_model))

    controller = let scontroller = SEDL.simulation_controller(sce)
        (args...) -> begin
            control = scontroller(args...)
            SEDL.BatchTuple(tconf, [map(v -> [v], control)])
        end
    end

    test_x0_batch = SEDL.BatchTuple(tconf, [sample_x0() for _ in 1:n_test_ex])
    x0_batch = SEDL.BatchTuple(tconf, [sample_x0() for _ in 1:n_train_ex])

    sample_next_state = (x -> x.next_state) ∘ motion_model
    sample_observation = rand ∘ obs_model

    data_train = SEDL.simulate_trajectory(
        times, x0_batch, sample_next_state, sample_observation, controller
    )
    data_test = SEDL.simulate_trajectory(
        times, test_x0_batch, sample_next_state, sample_observation, controller
    )
    (merge(data_train, (; Δt=tconf(Δt))), merge(data_test, (; Δt=tconf(Δt))))
end

function data_from_source(src::RealData, tconf::TensorConfig)
    (; train_data_path, test_data_path) = src

    data_train = read_data_from_csv(train_data_path, tconf)
    data_test = read_data_from_csv(test_data_path, tconf)
    Δt1 = tconf(data_train.times[2] - data_train.times[1])
    Δt2 = tconf(data_test.times[2] - data_test.times[1])
    (merge(data_train, (; Δt=Δt1)), merge(data_test, (; Δt=Δt2)))
end