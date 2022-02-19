using PrettyTables
using DataFrames
using CSV
using Measurements
using StatsPlots
StatsPlots.default(; dpi=300, legend=:outerbottom)

function show_baseline_comparison(
    scenarios_and_csv::Vector{Tuple{String,String}},
    metric::String,
    method_names::Vector{String};
    backend,
    drop_uncertainty=true,
    lower_is_better=true,
)
    @assert drop_uncertainty "current only drop_uncertainty=true is supported"

    sce_results = map(scenarios_and_csv) do (_, csv_file)
        table = CSV.read(csv_file, DataFrame)
        # get the column corresponding to the metric
        scores = parse_measurement_mean.(table[:, metric])
        keys = table[:, "method"]
        d = Dict(zip(keys, scores))
        [get(d, k, missing) for k in method_names]
    end

    best_ids = map(enumerate(sce_results)) do (col, scores)
        find_f = lower_is_better ? findmin : findmax
        _, row = find_f(scores)
        (row, col)
    end
    hl_best = if backend == :text
        Highlighter((data, i, j) -> (i, j - 1) in best_ids; bold=true, foreground=:green)
    else
        LatexHighlighter((data, i, j) -> (i, j - 1) in best_ids, ["textbf"])
    end

    scenarios = getindex.(scenarios_and_csv, 1)

    table_header = ["method vs. data"; scenarios]
    table_data = hcat(method_names, sce_results...)
    pretty_table(
        table_data; backend, header=table_header, highlighters=hl_best, alignment=:l
    )
end

function parse_measurement_mean(text::AbstractString)
    segs = split(text, "±")
    @assert length(segs) == 2
    parse(Float64, segs[1])
end

methods_of_interest = [
    "Handwritten",
    "FitTV",
    "FitHand",
    "SVI",
    "EM",
    # "FitTruth",
]

function print_baseline_tables(backend=:text)
    data_paths = [
        ("Hover", "results/comparisons/Hover/average.csv"),
        ("HoverS", "results/comparisons/HoverS/average.csv"),
        ("Hover256", "results/comparisons/Hover256/average.csv"),
        ("Car", "results/comparisons/Car/average.csv"),
        ("CarS", "results/comparisons/CarS/average.csv"),
    ]

    println("==== State estimation RMSE ====")
    show_baseline_comparison(data_paths, "RMSE", methods_of_interest; backend)
    println("==== Observation Log Probability ====")
    show_baseline_comparison(
        data_paths, "log_obs", methods_of_interest; lower_is_better=false, backend
    )
end

# FIXME: outdated
function print_perf_vs_schedule(backend=:text)
    data_paths = [
        ("σ=1°", "results/obs_schedule_variation_1.0.csv"),
        ("σ=5°", "results/obs_schedule_variation_5.0.csv"),
    ]

    println("==== State estimation RMSE ====")
    show_baseline_comparison(data_paths, "RMSE", methods_of_interest; backend)
    println("==== Forward prediction RMSE ====")
    show_baseline_comparison(data_paths, "open_loop", methods_of_interest; backend)
end

function print_perf_vs_noise(backend=:text)
    table = CSV.read("results/obs_noise_variation.csv", DataFrame)
    RMSE = parse_measurement_mean.(table[:, "RMSE"])
    open_loop = parse_measurement_mean.(table[:, "open_loop"])
    σs = map(table[:, "name"]) do name
        split(name, "=")[2]
    end
    println("==== Performance vs. observation noise ====")
    pretty_table(
        [σs RMSE open_loop];
        header=["σ_bearing", "state est.", "forward pred."],
        alignment=:l,
        backend,
    )
end

function plot_perf_vs_noise(; plot_args...)
    method_names = [
        "Handwritten" => "Handwritten",
        "FitTV" => "FitTV",
        "FitHand" => "FitHand",
        "SVI" => "SVI",
        "EM" => "STEADY",
        "FitTruth" => "FitTruth",
    ]
    data_files = map([1, 4, 8, 12, 16]) do deg
        deg => "results/vary_obs_noise/$(deg)°.csv"
    end

    xs = getindex.(data_files, 1)
    errors = map(data_files) do (_, file)
        table = CSV.read(file, DataFrame)
        methods = table[:, "name"]
        RMSE = parse_measurement_mean.(table[:, "RMSE"])
        Dict(zip(methods, RMSE))
    end

    plt = plot(;
        xlabel="bearing σ (in degree)",
        ylabel="Posterior RMSE",
        legend=:topleft,
        plot_args...,
    )
    for (method, method_name) in method_names
        ys = [d[method] for d in errors]
        extra_args = if method == "FitTruth"
            [:linestyle => :dash, :markershape => :none]
        else
            []
        end
        plot!(xs, ys; label=method_name, markershape=:auto, markersize=3, extra_args...)
    end
    plt
end
##-----------------------------------------------------------

print_baseline_tables()
print_baseline_tables(:latex)
print_perf_vs_schedule()
print_perf_vs_noise()
let plt = plot_perf_vs_noise(; size=(450, 300), xticks=[1, 4, 8, 12, 16])
    display(plt)
    savefig(plt, "results/vary_obs_noise/vary_obs_noise.pdf")
end

StatsPlots.plotattr(:Series)