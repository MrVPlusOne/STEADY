using StatsPlots
StatsPlots.default(; dpi=300, legend=:outerbottom)
include("../src/SEDL.jl")  # reloads the module
using .SEDL: SEDL
##-----------------------------------------------------------
experiment_results = SEDL.run_simulation_experiments(; is_test_run=false)
for (name, r) in experiment_results
    println("------- Scenario: $name -------")
    display(r)
end
##-----------------------------------------------------------