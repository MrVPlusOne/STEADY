export specific_elems, count_len, @unzip, @unzip_named
export max_by, sort_by
export rotate2d, rotation2D, °
export Optional, map_optional

using MacroTools: @capture
using ForwardDiff: Dual
# import ReverseDiff
using LineSearches: LineSearches

const Optional{X} = Union{X,Nothing}
const AbsVec = AbstractVector
const AbsMat = AbstractMatrix

abstract type Either end

struct Left{X} <: Either
    value::X
end

struct Right{Y} <: Either
    value::Y
end

specific_elems(xs::AbstractArray{T}) where {T} = Base.isconcretetype(T) ? xs : identity.(xs)

count_len(iters) = count(_ -> true, iters)

get_columns(m::Matrix) = (m[:, i] for i in 1:size(m, 2))
get_rows(m::Matrix) = (m[i, :] for i in 1:size(m, 1))

"""
Like @assert, but try to print out additional information about the arguments.
## Examples
```julia
julia> let a=4, b=2; @smart_assert(a < b, "Some helpful info.") end

ERROR: LoadError: AssertionError: Some helpful info. | Caused by: Condition `a < b` failed due to `a` => 4, `b` => 2 .
Stacktrace:
...
```
"""
macro smart_assert(ex, msg=nothing)
    has_msg = msg !== nothing
    if @capture(ex, op_(lhs_, rhs_))
        ex_q = QuoteNode(ex)
        lhs_q = QuoteNode(lhs)
        rhs_q = QuoteNode(rhs)
        quote
            lv = $(esc(lhs))
            rv = $(esc(rhs))
            if !$(esc(op))(lv, rv)
                reason_text = "Condition `$($ex_q)` failed due to `$($lhs_q)` => $lv, `$($rhs_q)` => $rv ."
                if $has_msg
                    msg_v = $(esc(msg))
                    throw(AssertionError("$msg_v | Caused by: $reason_text"))
                else
                    throw(AssertionError(reason_text))
                end
            end
        end
    else
        if has_msg
            esc(:(@assert($ex, $msg)))
        else
            esc(:(@assert($ex)))
        end
    end
end

function normalize_transform(xs::AbstractVector)
    σ = std(xs)
    μ = mean(xs)
    σ = map(σ) do s
        s ≈ zero(s) ? one(s) : s
    end
    transformed = map(xs) do v
        (v .- μ) ./ σ
    end
    (; transformed, μ, σ)
end

"""
Concat columns horizontally.
"""
hcatreduce(xs::AbsVec) = reduce(hcat, xs)
"""
Concat rows vertically.
"""
vcatreduce(xs::AbsVec) = reduce(vcat, xs)

"""
    @unzip xs, [ys,...] = collection

## Example
```jldoctest
julia> @unzip as, bs = [("a", "b", "c") for i in 1:3]
3-element Vector{Tuple{String, String, String}}:
 ("a", "b", "c")
 ("a", "b", "c")
 ("a", "b", "c")

julia> as
3-element Vector{String}:
 "a"
 "a"
 "a"

julia> bs
3-element Vector{String}:
 "b"
 "b"
 "b"
```
"""
macro unzip(assign)
    if @capture(assign, (v1_, vs__) = rhs_)
        assigns = map(enumerate([v1; vs])) do (i, v)
            :($(esc(v)) = map(x -> x[$i], rhs_value))
        end
        Expr(:block, :(rhs_value = $(esc(rhs))), assigns..., :rhs_value)
    else
        error("Usage: `@unzip x, [y, ...] = rhs`")
    end
end

"""
## Example
```jldoctest
julia> @unzip_named (as, :a), (cs, :c) = [(a="a", b="b", c="c") for i in 1:3]
3-element Vector{NamedTuple{(:a, :b, :c), Tuple{String, String, String}}}:
 (a = "a", b = "b", c = "c")
 (a = "a", b = "b", c = "c")
 (a = "a", b = "b", c = "c")

julia> as
3-element Vector{String}:
 "a"
 "a"
 "a"

julia> cs
3-element Vector{String}:
 "c"
 "c"
 "c"
```
"""
macro unzip_named(assign)
    fail() = error("Usage: `@unzip_named (xs, :x_name), [(ys, :y_name), ...] = rhs`")

    @capture(assign, (v1_, vs__) = rhs_) || fail()
    assigns = map([v1; vs]) do v
        (v isa Expr && v.head === :tuple && length(v.args) == 2) || fail()
        x, x_name = v.args
        (x isa Symbol && x_name isa QuoteNode) || fail()
        :($(esc(x)) = map(p -> getfield(p, $x_name), rhs_value))
    end
    Expr(:block, :(rhs_value = $(esc(rhs))), assigns..., :rhs_value)
end


@inline rotation2D(θ) = @SArray([
    cos(θ) -sin(θ)
    sin(θ) cos(θ)
])

rotate2d(θ, v::AbstractArray) = rotation2D(θ) * v

const ° = π / 180

to_measurement(values::AbsVec{<:Real}) = begin
    @assert !isempty(values)
    μ = mean(values)
    σ = std(values)
    μ ± σ
end

to_measurement(values::AbsVec{<:AbstractDict}) = begin
    @assert !isempty(values)

    keyset = Set(keys(values[1]))

    for d in values
        @smart_assert keyset == Set(keys(d))
    end

    (key => to_measurement(map(x -> x[key], values)) for key in keyset) |> Dict
end

"""
```jldoctest
julia> to_measurement([(a=1,b=2),(a=3,b=6)])
(a = 2.0 ± 1.4, b = 4.0 ± 2.8)
```
"""
to_measurement(values::AbstractVector{<:NamedTuple}) = begin
    vec_values = map(structure_to_vec, values)
    template = values[1]
    μ = mean(vec_values)
    σ = std(vec_values)
    structure_from_vec(template, μ .± σ)
end

pretty_number(v) = (v isa Number ? format(v; commas=true) : string(v))
pretty_number(v::Measurement) = string(v)

"""
Apply a tuple of functions to a tuple of corresponding arguments. The result is also a 
tuple.

Currently, the result type is restricted to be the same type as `xs` to aid type inference
when `length(xs)` is large.
"""
@inline function zipmap(fs, xs::X)::X where {X<:Tuple}
    @assert length(fs) == length(xs) "Need the same number of functions and values"
    ntuple(i -> fs[i](xs[i]), length(xs))
end

"""
Apply a tuple of functions to a NamedTuple of corresponding arguments. The result is a 
NamedTuple.
"""
@inline function zipmap(fs, xs::X)::X where {X<:NamedTuple}
    @smart_assert length(fs) == length(xs) "Need the same number of functions and values.\
        \nfs = $fs\nxs = $xs."
    t = ntuple(i -> fs[i](xs[i]), length(xs))
    NamedTuple{keys(xs)}(t)
end

function zipmap(fs, xs::AbstractVector)
    @smart_assert length(fs) == length(xs) "Need the same number of functions and values"
    map((f, x) -> f(x), fs, xs)
end

subtuple(xs::NamedTuple, keys::Tuple) = begin
    NamedTuple{keys}(map(k -> getfield(xs, k), keys))
end

"""
Warp the given angle into the range [0, 2π].
"""
warp_angle(angle::Real) =
    let
        x = angle % 2π
        x < 0 ? x + 2π : x
    end

angular_distance(θ1, θ2) = begin
    Δ = abs(θ1 - θ2) % 2π
    min(Δ, 2π - Δ)
end

"""
Lightweight version of @timed.
"""
macro ltimed(ex)
    quote
        t0 = time()
        v = $(esc(ex))
        (time=time() - t0, value=v)
    end
end

"""
Drop the gradient type tag to reduce JIT compilation time and avoid tag checking.
"""
function optimize_no_tag(loss, x₀, optim_options)
    cfg = ForwardDiff.GradientConfig(nothing, x₀)
    function fg!(F, G, x)
        (G === nothing) && return loss(x)

        gr = ForwardDiff.DiffResult(first(x), (G,))
        ForwardDiff.gradient!(gr, loss, x, cfg)
        if F !== nothing
            return gr.value
        end
    end
    algorithm = Optim.LBFGS(; linesearch=LineSearches.BackTracking(; order=3))
    Optim.optimize(Optim.only_fg!(fg!), x₀, algorithm, optim_options)
end

# """
# Use reverse mode autodiff to optimize the loss.
# """
# function optimize_reverse_diff(loss, x₀, optim_options)
#     # drop tag to reduce JIT compilation time and avoid tag checking
#     cfg = ReverseDiff.GradientConfig(x₀) 
#     f_tape = ReverseDiff.GradientTape(loss, x₀, cfg)
#     compiled_tape = ReverseDiff.compile(f_tape)
#     function fg!(F, G, x)
#         (G === nothing) && return loss(x)

#         gr = ReverseDiff.DiffResult(first(x), (G,))
#         ReverseDiff.gradient!(gr, compiled_tape, x)
#         if F !== nothing
#             return gr.value
#         end
#     end
#     algorithm = Optim.LBFGS(linesearch=LineSearches.BackTracking(order=3))
#     Optim.optimize(Optim.only_fg!(fg!), x₀, algorithm, optim_options)
# end

function optimize_bounded(loss, x₀, (lower, upper), optim_options)
    # drop tag to reduce JIT compilation time and avoid tag checking
    cfg = ForwardDiff.GradientConfig(nothing, x₀)
    function fg!(F, G, x)
        (G === nothing) && return loss(x)

        gr = ForwardDiff.DiffResult(first(x), (G,))
        ForwardDiff.gradient!(gr, loss, x, cfg)
        if F !== nothing
            return gr.value
        end
    end
    optimizer = Optim.Fminbox(Optim.LBFGS())
    Optim.optimize(Optim.only_fg!(fg!), lower, upper, x₀, optimizer, optim_options)
end

# function optimize_no_tag(loss, x₀, optim_options)
#     Optim.optimize(loss, x₀, Optim.LBFGS(), optim_options, autodiff=:forward)
# end

to_svec(vec::AbstractVector) = SVector{length(vec)}(vec)
to_svec(vec::NTuple{n,X}) where {n,X} = SVector{n,X}(vec)

to_static_array(array::AbsMat) = SMatrix{size(array)...}(array)
to_static_array(array::AbsVec) = SVector{length(array)}(array)
"""
Like `get!`, but can be used to directly access nested dictionaries.

```jldoctest
julia> d = Dict{Int, Dict{Symbol, String}}()
Dict{Int64, Dict{Symbol, String}}()

julia> nested_get!(d, 5 => :a) do
           "default"
       end
"default"

julia> d
Dict{Int64, Dict{Symbol, String}} with 1 entry:
  5 => Dict(:a=>"default")
```
"""
function nested_get!(f, d::Dict{K,D}, k_pair::Pair{K,Rest}) where {K,D<:Dict,Rest}
    inner_dict = get!(d, k_pair[1]) do
        D()
    end
    nested_get!(f, inner_dict, k_pair[2])
end

function nested_get!(f, d::Dict{K,V}, k::K) where {K,V}
    get!(f, d, k)
end

function is_bad_dual(v::Dual)
    isfinite(v.value) && abs(v.value) < 1e5 && any(isnan, v.partials)
end
is_bad_dual(v::AbstractVector) = any(is_bad_dual, v)
is_bad_dual(v::Real) = false

"""
Convert any `NaN` partial derivatives to zeros. 
"""
fix_nan_dual(v::Dual{Tag}) where {Tag} =
    let
        if is_bad_dual(v)
            new_partials = ForwardDiff.Partials(nan_to_zero.(v.partials.values))
            Dual{Tag}(v.value, new_partials)
        else
            v
        end
    end
fix_nan_dual(v::Real) = v

function nan_to_zero(v::R)::R where {R<:Real}
    isnan(v) ? zero(v) : v
end

using Statistics: norm
Base.show(io::IO, d::ForwardDiff.Dual) =
    print(io, "Dual($(d.value), |dx|=$(norm(d.partials)))")

function assert_finite(x::NamedTuple)
    if !all(isfinite, x)
        @error "some components are not finite" x
        throw(ErrorException("assert_finite failed."))
    end
    x
end

"""
A helper macro to find functions by name.
"""
@generated function find_func(T, args...)
    return Expr(:splatnew, :T, :args)
end

"""
Numerical integration based on the leap-frog step.
`a_f(x, x′) -> a` is the acceleration function.
"""
function leap_frog_step((x, v, a)::Tuple{X,X,X}, a_f, Δt) where {X}
    v_half = @. v + (Δt / 2) * a
    x1 = @.(x + Δt * v_half)::X
    a1 = a_f(x1, @.(v_half + (Δt / 2) * a))::X
    v1 = @.(v + (Δt / 2) * (a + a1))::X
    (x1, v1, a1)
end

##-----------------------------------------------------------
# convenient combinators
Base.map(f::Function) = xs -> map(f, xs)
Base.filter(f::Function) = xs -> filter(f, xs)
max_by(f::Function) = xs -> begin
    ys = map(f, xs)
    _, i = findmax(ys)
    xs[i]
end

sort_by(f::Function) = xs -> sort(xs; by=f)

"""
    map_optional(f, x) = x === nothing ? nothing : f(x)
"""
map_optional(f::Function, ::Nothing) = nothing
map_optional(f::Function, x) = f(x)
##-----------------------------------------------------------
using Distributed
using ProgressMeter: Progress, next!

const _distributed_work_ctx = Ref{Any}(nothing)

"""
# Arguments
- `f(ctx, x)`: the context-dependent task to be performed.
"""
function parallel_map(
    f,
    xs,
    ctx;
    progress::Progress=Progress(1; enabled=false),
    n_threads::Integer=Threads.nthreads(),
    use_distributed::Bool=false,
)
    if use_distributed
        length(workers()) > 1 || error("distributed enabled with less than 2 workers.")
        @everywhere SEDL._distributed_work_ctx[] = $ctx
        f_task = x -> let ctx = _distributed_work_ctx[]
            try
                Right(f(ctx, x))
            catch e
                Left((e, stacktrace(catch_backtrace())))
            end
        end
        try
            map_results = progress_pmap(f_task, xs; progress)
        finally
            @everywhere SEDL._distributed_work_ctx[] = nothing
        end
    else
        eval_task(e) =
            try
                let r = f(ctx, e)
                    next!(progress)
                    Right(r)
                end
            catch e
                Left((e, stacktrace(catch_backtrace())))
            end
        if n_threads <= 1
            map_results = map(eval_task, xs)
        else
            pool = ThreadPools.QueuePool(2, n_threads)
            try
                map_results = ThreadPools.tmap(eval_task, pool, xs)
            finally
                close(pool)
            end
        end
    end
    eval_results = if all(x -> x isa Right, map_results)
        map(x -> x.value, map_results)
    else
        for x in map_results
            if x isa Left
                (e, trace) = x.value
                @error "Exception in parallel_map: " expection = e
                @error "Caused by: $(repr("text/plain", trace))"
                error("parallel_map failed.")
            end
        end
    end
    eval_results
end

sigmoid(x::Real) = one(x) / (one(x) + exp(-x))

"""
Combine a named tuple of functions into a function that returns a named tuple.
"""
function combine_functions(functions::NamedTuple{names}) where {names}
    x -> NamedTuple{names}(map(f -> f(x), values(functions)))
end

function data_dir(segs...)
    path = read(projectdir("configs/datadir.txt"), String) |> strip
    joinpath(path, segs...)
end