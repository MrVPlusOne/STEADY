using PrettyTables
using DataFrames
using CSV
using Measurements
using SmartAsserts
using StatsPlots
using Printf
using Statistics: mean
StatsPlots.default(; dpi=300, legend=:outerbottom)

MethodRenamingTable = [
    "Super_Hand" => "FitHand",
    "Super_TV" => "FitTV",
    "Super_noiseless" => "FitTruth",
    "VI" => "SVI",
]

MethodsOfInterest = [
    "Handwritten",
    "FitHand",
    "FitTV",
    "SVI",
    "EM",
    "FitTruth",
]

function show_baseline_comparison(
    scenarios::Vector{String},
    tables::Vector{<:DataFrame},
    metric::String,
    method_names::Vector{String}=MethodsOfInterest;
    backend,
    drop_uncertainty=true,
    lower_is_better=true,
    number_format="%.3f",
)
    @assert drop_uncertainty "current only drop_uncertainty=true is supported"

    sce_results = map(tables) do table
        # get the column corresponding to the metric
        scores = if metric == "pose" || metric == "fw_pose"
            prefix = metric == "pose" ? "" : "fw_"
            error_loc = parse_measurement_mean.(table[:, "$(prefix)location"])
            error_angle = parse_measurement_mean.(table[:, "$(prefix)angle"])
            @. sqrt(error_loc ^ 2 + error_angle ^ 2)
        else
            parse_measurement_mean.(table[:, metric])
        end
        keys = replace(table[:, "method"], MethodRenamingTable...)
        d = Dict(zip(keys, scores))
        [get(d, k, missing) for k in method_names]
    end

    best_ids = map(enumerate(sce_results)) do (col, scores)
        if lower_is_better
            _, row = findmin(replace(scores[1:end-1], missing => Inf))
        else
            _, row = findmax(replace(scores[1:end-1], missing => -Inf))
        end
        (row, col)
    end
    hl_best = if backend == :text
        Highlighter((data, i, j) -> (i, j - 1) in best_ids; bold=true, foreground=:green)
    else
        LatexHighlighter((data, i, j) -> (i, j - 1) in best_ids, ["textbf"])
    end

    table_header = ["method vs. data"; scenarios]
    fmt = Printf.Format(number_format)
    number_strings = map(sce_results) do nums 
        [n isa Number ? Printf.format(fmt, n) : string(n) for n in nums]
    end
    table_data = hcat(method_names, number_strings...)
    pretty_table(
        table_data; backend, header=table_header, highlighters=hl_best, alignment=:l
    )
end

function parse_measurement(text::AbstractString)
    @smart_assert occursin("±", text)
    segs = split(text, "±")
    @smart_assert length(segs) == 2
    μ = parse(Float64, segs[1])
    σ = parse(Float64, segs[2])
    μ ± σ
end

function parse_measurement_mean(text::AbstractString)
    if occursin("±", text)
        segs = split(text, "±")
        @smart_assert length(segs) == 2
        parse(Float64, segs[1])
    else
        parse(Float64, text)
    end
end
function parse_measurement_mean(n::AbstractFloat)
    convert(Float64, n)
end

function print_baseline_tables(; backend=:text, aggregate="average")
    data_paths = [
        ("Hover", "reports/comparisons/hovercraft/"),
        ("Hover160", "reports/comparisons/hovercraft160/"),
        # ("Car", "reports/comparisons/ut_automata/"),
        ("Truck", "reports/comparisons/alpha_truck/"),
    ]
    scenarios = getindex.(data_paths, 1)
    tables = map(data_paths) do (_, path)
        CSV.read(joinpath(path, "$aggregate.csv"), DataFrame)
    end

    # println("==== State Estimation RMSE (average) ====")
    # show_baseline_comparison(scenarios, avg_tables, "RMSE"; backend)
    # println("==== State Estimation RMSE ($aggregate) ====")
    # show_baseline_comparison(scenarios, tables, "total"; backend)
    println("==== State Estimation location RMSE ($aggregate) ====")
    show_baseline_comparison(scenarios, tables, "location"; backend)
    println("==== State Estimation orientation RMSE ($aggregate) ====")
    show_baseline_comparison(scenarios, tables, "angle"; backend)

    # println("==== Forward Prediction RMSE ($aggregate) ====")
    # show_baseline_comparison(scenarios, tables, "fw_total"; backend)
    println("==== Forward Prediction location RMSE ($aggregate) ====")
    show_baseline_comparison(scenarios, tables, "fw_location"; backend)
    println("==== Forward Prediction orientation RMSE ($aggregate) ====")
    show_baseline_comparison(scenarios, tables, "fw_angle"; backend)
end


function plot_perf_vs_noise(; plot_args...)
    data_dirs = map([1.25, 2.5, 5.0, 10.0, 20.0]) do deg
        deg => "reports/obs_noise_variation/$(deg)°"
    end

    xs = getindex.(data_dirs, 1)
    ys_val = map(data_dirs) do (_, dir)
        table = CSV.read(joinpath(dir, "average.csv"), DataFrame)
        methods = table[:, "method"]
        RMSE = getproperty.(parse_measurement.(table[:, "fw_total"]), :val)
        Dict(zip(methods, RMSE))
    end
    ys_err = map(data_dirs) do (_, dir)
        table = CSV.read(joinpath(dir, "average.csv"), DataFrame)
        methods = table[:, "method"]
        RMSE = getproperty.(parse_measurement.(table[:, "fw_total"]), :err)
        Dict(zip(methods, RMSE))
    end

    plt = plot(;
        xlabel="bearing σ (in degree)",
        ylabel="Forward Prediction Error",
        legend=:topleft,
        plot_args...,
    )

    methods = insert!(copy(MethodsOfInterest), length(MethodsOfInterest)-1, "EM_NS")
    method_rename = Dict("EM" => "STEADY", "EM_NS" => "STEADY-")
    for method in methods
        if method in ["FitTruth", "Handwritten"]
            # average the results for these noise-independent baselines
            y = mean([get(d, method, missing) for d in ys_val])
            ys = fill(y, length(ys_val))
            ribbon = nothing
        else
            ys = [get(d, method, missing) for d in ys_val]
            ribbon = [get(d, method, missing) for d in ys_err]
        end
        extra_args = if method == "FitTruth"
            [:linestyle => :dash, :markershape => :none]
        else
            []
        end
        label = get(method_rename, method, method)
        plot!(xs, ys; label, markershape=:auto, markersize=3, ribbon, extra_args...)
    end
    plt
end

function plot_training_curve_vs_particles(; x_is_time=true, plt_args...)
    plt = plot(; plt_args...)
    particle_sizes = [200, 2_000, 20_000, 200_000]
    for n_particle in particle_sizes
        label = "n_particle=$(n_particle/1000)K"
        curve = CSV.read(joinpath("reports/vary_particle_size/", "$label.csv"), DataFrame)
        steps = curve[:, :step]
        times = curve[:, :training_time]
        ys = curve[:, :total]
        if x_is_time
            times = times .- times[1] # remove the initialization time.
            ids = filter(i -> 400 <= times[i] <= 5000, eachindex(steps))
            xs = times[ids]
            xlabel="training time (s)"
        else
            ids = filter(i -> 3_000 <= steps[i] <= 30_000, eachindex(steps))
            xs = steps[ids]
            xlabel="training steps"
        end
        plot!(
            plt,
            xs,
            ys[ids];
            xlabel,
            ylabel="State Estimation Error",
            label,
        )
    end
    plt
end
##-----------------------------------------------------------
print_baseline_tables(aggregate="best")
print_baseline_tables(aggregate="best", backend=:latex)
let xticks=[1.25, 2.5, 5, 10, 20]
    plt = plot_perf_vs_noise(; legend=:top, size=(500, 350), xticks=(xticks, map(string, xticks)), xscale=:log)
    display(plt)
    savefig(plt, "reports/obs_noise_variation/vary_obs_noise.pdf")
end

let plt = plot_training_curve_vs_particles(;
        x_is_time=true,
        legend=:topright, size=(450, 300), xticks=[400, 1000, 2000, 3000, 4000]
    )
    display(plt)
    savefig(plt, "reports/vary_particle_size/vary_particle_size.pdf")
end