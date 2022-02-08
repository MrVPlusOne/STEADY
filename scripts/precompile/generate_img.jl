pkg_list = [
    :BenchmarkTools,
    :Bijectors,
    :CUDA,
    :DataFrames,
    :DataStructures,
    :ProgressMeter,
    :ReverseDiff,
    :RuntimeGeneratedFunctions,
    :StaticArrays,
    :Statistics,
    :StatsBase,
    :StatsFuns,
    :StatsPlots,
    :SymbolicRegression,
    :TensorBoardLogger,
    :ThreadPools,
    :Zygote,
]

using PackageCompiler
println("Creating Sysimage...")
time_taken = @elapsed create_sysimage(
    pkg_list,
    sysimage_path="JuliaSysimage.so",
    precompile_execution_file="scripts/precompile/workload.jl",
)
@show time_taken