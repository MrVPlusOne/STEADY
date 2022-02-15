!true && begin
    include("../../src/SEDL.jl")
    using .SEDL
end
using SEDL
using Alert

dynamic_include = include # to avoid mess up the VSCode linter

function run_multiple_times(run_args, n_repeats)
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