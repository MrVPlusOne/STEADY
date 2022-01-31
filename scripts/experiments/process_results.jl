using PrettyTables
using DataFrames
using CSV
using Measurements

function show_baseline_comparison(
    scenarios_and_csv::Vector{Tuple{String,String}},
    metric::String;
    drop_uncertainty=true,
    lower_is_better=true,
)
    @assert drop_uncertainty "current only drop_uncertainty=true is supported"

    method_names = nothing

    sce_results = map(scenarios_and_csv) do (_, csv_file)
        table = CSV.read(csv_file, DataFrame)
        # get the column corresponding to the metric
        scores = parse_measurement.(table[:, metric])
        method_names === nothing && (method_names = table[:, "name"])
        d = Dict(zip(method_names, scores))
        [d[n] for n in method_names]
    end

    best_ids = map(enumerate(sce_results)) do (col, scores)
        find_f = lower_is_better ? findmin : findmax
        _, row = find_f(scores)
        (row, col)
    end
    hl_best = Highlighter((data, i, j) -> (i, j-1) in best_ids; bold=true, foreground=:green)

    scenarios = getindex.(scenarios_and_csv, 1)

    table_header = ["method \\ data"; scenarios]
    table_data = hcat(method_names::Vector, sce_results...)
    pretty_table(table_data; header=table_header, highlighters=hl_best, alignment=:l)
end

function parse_measurement(text::AbstractString)
    segs = split(text, "±")
    @assert length(segs) == 2
    parse(Float64, segs[1])
end

data_paths = [
    ("Hover", "results/comparisons-hovercraft.csv"),
    ("Hover-Fixed", "results/comparisons-hovercraft-fixed_var.csv"),
    ("Hover-Gaussian", "results/comparisons-hovercraft-gaussian.csv"),
    ("RealCar", "results/comparisons-real.csv"),
]

println("==== State estimation RMSE ====")
show_baseline_comparison(data_paths, "RMSE")
println("==== Forward prediction RMSE ====")
show_baseline_comparison(data_paths, "open_loop")
