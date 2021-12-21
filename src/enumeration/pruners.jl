"""
Each prunner can optionally override the following functions
- [`prune_iteration!`](@ref)
- [`prune_immediately!`](@ref)
- [`reset!`](@ref)
"""
abstract type AbstractPruner end

"""
Should return the list of programs to be pruned, each entry of the form `(pruned, reason)`.

If `types_to_prune` is empty, will prune expressions of all types.
"""
prune_iteration!(::AbstractPruner, ::EnumerationResult, types_to_prune, size; is_last) = []

"""
Reset the state of the prunner. This is called at the start of each synthesis enumeration.
"""
reset!(::AbstractPruner) = nothing

"""
Should return either nothing if should not prune or the pruning reason.
"""
prune_immediately!(::AbstractPruner, prog::TAST, size) = nothing

"""
Does no prunning at all.
"""
struct NoPruner <: AbstractPruner end


"""
Prune programs using their input-output behaviors on a given set of inputs. 
"""
@kwdef(
    struct IOPruner{In} <: AbstractPruner
        inputs::Vector{In}
        comp_env::ComponentEnv
        tolerance::Float64 = 1e-10
        "how many digits to use when rounding values in the IO vectors."
        tol_digits::Int = floor(Int, -log10(tolerance))
        prog_to_vec::Dict{TAST,Vector{<:Any}} = Dict{TAST,Vector{<:Any}}()
        vec_to_prog::Dict{Tuple{PType,Vector{<:Any}},TAST} = Dict{
            Tuple{PType,Vector{<:Any}},TAST
        }()
    end
)

function round_values(v::Real; digits)
    round(v; digits)
end

function round_values(v::AbstractVector; digits)
    round.(v; digits)
end

function is_valid_value(v::Union{Real,AbstractVector})
    isfinite(v)
end

function reset!(pruner::IOPruner)
    empty!(pruner.prog_to_vec)
    empty!(pruner.vec_to_prog)
end

"""
Return either nothing if should not prune or the pruning reason.
"""
function prune_immediately!(pruner::IOPruner, prog::TAST, size::Integer)
    (; inputs, comp_env, prog_to_vec, vec_to_prog, tol_digits) = pruner
    local new_vec
    @assert !haskey(prog_to_vec, prog)
    if prog isa Call
        arg_vecs = map(prog.args) do a
            prog_to_vec[a]
        end
        f = comp_env.impl_dict[prog.f]
        try
            new_vec = f.(arg_vecs...)
        catch e
            if e isa Union{OverflowError,DomainError}
                return "evaluation error: $e"
            else
                rethrow(e)
            end
        end
    else
        @assert prog isa Var
        new_vec = getfield.(inputs, prog.name)
    end
    all(is_valid_value, new_vec) || return "invalid values: $new_vec"

    vec_rounded = round_values(new_vec; digits=tol_digits)
    similar_p = get(vec_to_prog, (prog.type, vec_rounded), nothing)
    similar_p === nothing || return "similar program: $similar_p"

    prog_to_vec[prog] = new_vec
    vec_to_prog[(prog.type, vec_rounded)] = prog

    return nothing
end

total_time_report(reports) = begin
    tos = (r -> r.to).(reports)
    reduce(merge, tos)
end

default_saturation_params(egraph_size) = begin
    n = max(egraph_size, 500)
    # SaturationParams(
    #     scheduler=Metatheory.Schedulers.BackoffScheduler,
    #     schedulerparams=(n, 5),
    #     timeout=8, eclasslimit=n, enodelimit=4n, matchlimit=4n)
    SaturationParams(;
        threaded=true,
        scheduler=Metatheory.Schedulers.SimpleScheduler,
        timeout=2,
        eclasslimit=0,
        enodelimit=0,
        matchlimit=0,
    )
end