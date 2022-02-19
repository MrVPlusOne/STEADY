!true && begin
    include("../../src/SEDL.jl")
    using .SEDL
    using .SEDL: @kwdef, °
end
using SmartAsserts: @smart_assert
using SEDL
using Alert

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