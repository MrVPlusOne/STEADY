!true && begin
    include("../../src/SEDL.jl")
    using .SEDL
    using .SEDL: @kwdef, °
end
using SEDL
using SmartAsserts: @smart_assert
using Alert
using StatsPlots
StatsPlots.default(; dpi=300, legend=:outerbottom)

AllTrainingMethods = [
    # Use Handwritten models provided by each scenario, no learning performed.
    :Handwritten,
    # Use handwritten to estimate states, then learning from them.
    :FitHand,
    # Use MAP estimation to obtain states from observations, then apply total 
    # variation regularization to obtain learning data.
    :FitTV,
    # Directly learning from ground truth states. Does not use the observation model.
    :FitTruth,
    # Stochastic variational inference.
    :SVI,
    # Simultaneous state estimation and dynamics learning using EM.
    :EM,
    # Simultaneous SLAM and dynamics learning using EM.
    :EM_SLAM,
]

check_training_method(method::Symbol) = @smart_assert method ∈ AllTrainingMethods

dynamic_include = include # to avoid mess up the VSCode linter

function train_multiple_times(run_args, n_repeats)
    valid_perfs = []
    test_perfs = []
    for i in 1:n_repeats
        args = merge(run_args, (; run_id = i))
        @eval(Main, script_args = $args)
        dynamic_include("../train_models.jl")
        push!(valid_perfs, Main.valid_performance)
        push!(test_perfs, Main.test_performance)
    end
    local valid_performance = SEDL.named_tuple_reduce(valid_perfs, identity)
    local test_performance = SEDL.named_tuple_reduce(test_perfs, identity)
    (; valid_performance, test_performance)
end

function with_alert(task::Function, task_name::String, report_finish=true)
    try
        local result = task()
        if report_finish
            alert("(GPU=$(Main.GPU_ID)) $task_name finished.")
        end
        result
    catch e
        if e isa InterruptException
            throw(InterruptException()) # don't need the notification and stack trace.
        end
        alert("(GPU=$(Main.GPU_ID)) $task_name stopped due to exception: $(summary(e)).")
        rethrow()
    end
end

Default_Training_Args = (;
    scenario=SEDL.RealCarScenario("alpha_truck"),
    is_quick_test=false,
    load_trained=false,
    should_train_dynamics=true, # whether to train a NN motion model or use the ground truth
    gpu_id=nothing, # integer or nothing
    use_simple_obs_model=false,
    σ_bearing=5°,
    use_fixed_variance=false,
    train_method=:EM, # see `AllTrainingMethods`.
    n_train_ex=16,  # number of training trajectories when using simulation data
    validation_metric=:RMSE,  # :RMSE or :log_obs
    lr=1e-4,
    max_obs_weight=1.0,
    use_obs_weight_schedule=true, # whether to increase obs_weight from 0 to max_obs_weight over time
    max_train_steps=40_000,
    exp_name=nothing,
    n_particles=20_000,  # how many particles to use for the EM training.
    h_dim=64,
    run_id=1,
)

"""
Discard params that are the same as the default.
"""
function get_training_args_delta(args)
    changes = []
    foreach(keys(args)) do k
        @smart_assert k in keys(Default_Training_Args)
        if args[k] != Default_Training_Args[k]
            push!(changes, k => args[k])
        end
    end
    (; changes...)
end

function get_save_dir(script_args::NamedTuple)
    script_args = get_training_args_delta(script_args)
    config = merge(Default_Training_Args, script_args)
    prefix = config.is_quick_test ? "sims-quick" : "sims" 
    postfix = "run-$(config.run_id)"
    save_args = SEDL.dropnames(script_args, (:gpu_id, :is_quick_test, :run_id))
    SEDL.data_dir(
        prefix,
        savename("train_models-$(summary(config.scenario))", save_args; connector="-"),
        postfix,
    )
end