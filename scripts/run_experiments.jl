!true && include("../src/SEDL.jl")

using Revise
using StatsPlots
StatsPlots.default(; dpi=300, legend=:outerbottom)

using SEDL: SEDL

experiment_results = SEDL.run_simulation_experiments(; is_test_run=false)
for (name, r) in experiment_results
    println("-----", name, "-----")
    display(r)
end
##-----------------------------------------------------------