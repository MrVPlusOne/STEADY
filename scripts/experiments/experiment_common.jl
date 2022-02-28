!true && begin
    include("../../src/SEDL.jl")
    using .SEDL
    using .SEDL: @kwdef, °
end
using SEDL
using SmartAsserts: @smart_assert
using Alert
using DrWatson
using StatsPlots
StatsPlots.default(; dpi=300, legend=:outerbottom)

Default_Training_Args = (;
    exp_group="ungrouped",
    scenario=SEDL.RealCarScenario("alpha_truck"),
    is_quick_test=false,
    load_trained=false,
    should_train_dynamics=true, # whether to train a NN motion model or use the ground truth
    gpu_id=nothing, # integer or nothing
    use_simple_obs_model=false,
    σ_bearing=5°,
    use_fixed_variance=false,
    train_method=:EM, # see `AllTrainingMethods`.
    validation_metric=:log_obs,  # :RMSE or :log_obs
    lr=1e-4,
    max_obs_weight=1.0,
    use_obs_weight_schedule=true, # whether to increase obs_weight from 0 to max_obs_weight over time
    max_train_steps=40_000,
    n_particles=20_000,  # how many particles to use for the EM training.
    h_dim=64,
    run_id=1,
)

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
    :EM_NS,  # EM with no obs_w scheduling.
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
        time_taken = @elapsed begin
            result = task()
        end
        time_hours = time_taken / 3600
        msg = "(GPU=$(Main.GPU_ID), time_taken=$(time_hours)hours) $task_name finished."
        println(msg)
        if report_finish
            alert(msg)
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

"""
Discard params that are the same as the default.
"""
function get_modified_training_args(args)
    changes = []
    foreach(keys(args)) do k
        @smart_assert k in keys(Default_Training_Args)
        if args[k] != Default_Training_Args[k]
            push!(changes, k => args[k])
        end
    end
    (; changes...)
end

function get_save_dir(training_args::NamedTuple)
    modified = get_modified_training_args(training_args)
    config = merge(Default_Training_Args, modified)
    save_args = SEDL.dropnames(modified, (:gpu_id, :is_quick_test, :run_id, :exp_group))
    SEDL.data_dir(
        config.is_quick_test ? "sims-quick" : "sims",
        training_args.exp_group,
        savename(SEDL.filename(config.scenario), save_args; connector="-"),
        "run-$(config.run_id)",
    )
end