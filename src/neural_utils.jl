##-----------------------------------------------------------
using Flux: Flux
using Flux: Dense, Chain, LayerNorm, SkipConnection, BatchNorm
using Flux: @functor, softplus
using Adapt: Adapt
using Setfield: @set
using Zygote: Zygote
using Distributions: PDMat

export TensorConfig, tensor_type, check_type
"""
A configuration specifying which type (cpu or gpu) and floating type should be used
for array/tensor operations. 

## Usages
- Use `tconf(x)` to convert an object `x` (e.g. a number or an array) to having the 
desired configuration `tconf`.
- Use `check_type(tconf, x)` to check if the object `x` has the desired configuration.
"""
struct TensorConfig{on_gpu,ftype} end
TensorConfig(on_gpu, ftype=Float32) = TensorConfig{on_gpu,ftype}()

Base.show(io::IO, ::TensorConfig{on_gpu,ftype}) where {on_gpu,ftype} =
    print(io, (; on_gpu, ftype))

function Base.getproperty(::TensorConfig{on_gpu,ftype}, s::Symbol) where {on_gpu,ftype}
    if s === :on_gpu
        return on_gpu
    elseif s === :ftype
        return ftype
    else
        error("Unknown property: $s ")
    end
end

@inline function tensor_type(::TensorConfig{on_gpu,ftype}) where {on_gpu,ftype}
    (on_gpu ? CUDA.CuArray : Array){ftype}
end

new_array(tconf::TensorConfig{on_gpu,ftype}, sizes::Int...) where {on_gpu,ftype} =
    tensor_type(tconf)(undef, sizes...)

Adapt.adapt_storage(::Flux.FluxCPUAdaptor, tf::TensorConfig{true,ftype}) where {ftype} =
    TensorConfig{false,ftype}()

Adapt.adapt_storage(::Flux.FluxCUDAAdaptor, tf::TensorConfig{false,ftype}) where {ftype} =
    TensorConfig{true,ftype}()

# make sure it's not skipped by Flux.gpu
Flux._isbitsarray(::TensorConfig) = true

(tconf::TensorConfig{on_gpu,ftype})(v::Real) where {on_gpu,ftype} = convert(ftype, v)
(tconf::TensorConfig{on_gpu,ftype})(a::AbstractArray{<:Real}) where {on_gpu,ftype} =
    convert(tensor_type(tconf), a)

function check_type(tconf::TensorConfig{on_gpu,ftype}, x::Real) where {on_gpu,ftype}
    @smart_assert x isa ftype
end

function check_type(tconf::TensorConfig{on_gpu,ftype}, x) where {on_gpu,ftype}
    desired_type = tensor_type(tconf)
    @smart_assert typeof(x) <: desired_type
end

##-----------------------------------------------------------
export BatchTuple, common_batch_size, inflate_batch, split_components
"""
A wrapper over a batch of data with named components.

## Usages
Assuming `batch::BatchTuple`:
- Use `batch.val` to access the underlying NamedTuple, in which each 
component is a batch of data, with the last dimension having the batch size.
- Use `tconf(batch)` to change the batch to a different `TensorConfig`. 
- Use `batch[idx]` to access a subset of the batch, e.g., `tconf[1]` or `tconf[3:6]`, 
both returns a new `BatchTuple`.
- Use `inflate_batch(batch)` to expand the underlying value to full batch size (if the batch 
dimension is 1).
- Use `common_batch_size(batchs...)` to get the common batch size of a list of `BatchTuple`.
"""
struct BatchTuple{C<:TensorConfig,V<:NamedTuple}
    tconf::C
    batch_size::Int
    val::V
    BatchTuple(
        tconf::TensorConfig{on_gpu}, batch_size::Int, val::NamedTuple
    ) where {on_gpu} = begin
        foreach(keys(val)) do k
            v = getproperty(val, k)
            check_type(tconf, v)
            if v isa Real
                return nothing
            else
                @smart_assert(
                    ndims(v) >= 2,
                    "Batch values should have at least 2 dimensions, with the last one \
                    being the batch dimension."
                )
                size(v)[end] == 1 ||
                    size(v)[end] == batch_size ||
                    error("Element $k has size $(size(v)), but batch size is $batch_size")
            end
        end
        new{typeof(tconf),typeof(val)}(tconf, batch_size, val)
    end
end
Flux.@functor BatchTuple

function Base.show(io::IO, ::Type{T}) where {T<:BatchTuple}
    if T isa UnionAll || T.parameters[2] isa UnionAll
        print(io, "BatchTuple")
    else
        names = T.parameters[2].parameters[1]
        tconf = T.parameters[1]
        print(io, "BatchTuple{keys=$names, tconf=$tconf}")
    end
end

"""
Create from a batch of values.
"""
function BatchTuple(tconf::TensorConfig, values::Vector{<:NamedTuple})
    BatchTuple(tconf, length(values), map(tconf, hcatreduce(values)))
end

function BatchTuple(template::BatchTuple, val::NamedTuple)
    BatchTuple(template.tconf, template.batch_size, val)
end

function BatchTuple(batches::AbsVec{<:BatchTuple})
    BatchTuple(
        batches[1].tconf,
        sum(x -> x.batch_size, batches),
        hcatreduce((x -> inflate_batch(x).val).(batches))::NamedTuple,
    )
end

"""
Create a new BatchTuple from existing ones by combining their values
```
    BatchTuple(batch1, batch2...) do batch_val1, batch_val2...
        # should return new values
    end
````
"""
function BatchTuple(f::Function, batches::BatchTuple{TC}...) where {TC}
    bs = common_batch_size(batches...)
    BatchTuple(TC(), bs, f(getfield.(batches, :val)...)::NamedTuple)
end

Base.map(f, batch::BatchTuple) =
    BatchTuple(batch.tconf, batch.batch_size, map(f, batch.val))

function Base.map(f, batches::BatchTuple{TC}...) where {TC}
    bs = common_batch_size(batches...)
    BatchTuple(TC(), bs, map(f, map(b -> b.val, batches)...))
end

function Base.length(batch::BatchTuple)
    batch.batch_size
end

function Base.lastindex(batch::BatchTuple)
    batch.batch_size
end

function Base.getindex(batch::BatchTuple, ids)
    bs = batch.batch_size
    @smart_assert all(i -> 0 <= i <= bs, ids) "batch size = $(bs)"
    new_val = map(v -> batch_subset(v, ids), batch.val)
    BatchTuple(batch.tconf, length(ids), new_val)
end

"""
Split `batch` into `n` equally sized batches.
"""
function Base.split(batch::BatchTuple, n::Integer)
    @smart_assert batch.batch_size % n == 0
    chunksize = batch.batch_size ÷ n
    map(1:n) do i
        shift = 1 + (i - 1) * chunksize
        batch[shift:(shift + chunksize - 1)]
    end
end

"""
Similar to `merge(x1::NamedTuple, x2::NamedTuple)`.
"""
function Base.merge(b1::BatchTuple, b2::BatchTuple)
    @smart_assert b1.tconf == b2.tconf
    @smart_assert b1.batch_size == b2.batch_size
    BatchTuple(b1.tconf, b1.batch_size, merge(b1.val, b2.val))
end

function (tconf::TensorConfig)(batch::BatchTuple)
    if tconf == batch.tconf
        batch
    else
        new_val = map(tconf, batch.val)
        BatchTuple(tconf, batch.batch_size, new_val)
    end
end

function batch_subset(m::Union{Real,AbstractArray}, ids::Union{Integer,AbsVec{<:Integer}})
    r = if m isa Real
        m
    else
        s = size(m)
        if s[end] == 1
            m
        else
            colons = ntuple(_ -> :, ndims(m) - 1)
            ids1 = ids isa AbsVec ? ids : (ids:ids)
            m[colons..., ids1]
        end
    end
    r::typeof(m)
end
##-----------------------------------------------------------
# Batch utils
common_batch_size(sizes::Integer...) = begin
    max_size = max(sizes...)
    all(sizes) do s
        s == max_size || s == 1 || error("Inconsistent batch sizes: $sizes")
    end
    max_size
end

common_batch_size(bts::BatchTuple...) = begin
    common_batch_size(map(x -> x.batch_size, bts)...)
end

function Base.repeat(batch::BatchTuple, n::Int; inflate=false)
    nb = if n == 1
        batch
    elseif batch.batch_size == 1
        @set batch.batch_size = n
    else
        vs = map(inflate_batch(batch).val) do v
            repeat(v, ntuple(Returns(1), ndims(v) - 1)..., n)
        end
        BatchTuple(batch.tconf, batch.batch_size * n, vs)
    end
    inflate ? inflate_batch(nb) : nb
end

"""
Split a matrix into a named tuple of matrices according to the given sizes.

## Examples
```jldoctest
julia> split_components(ones(4, 5), (a=2, b=1, c=1))
(a = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], b = [1.0 1.0 … 1.0 1.0], c = [1.0 1.0 … 1.0 1.0])
```
"""
function split_components(x::AbsMat, comp_sizes::NamedNTuple{names,Int}) where {names}
    @smart_assert size(x, 1) == sum(comp_sizes)

    i = 1
    comps = map(names) do k
        x[i:((i += comp_sizes[k]) - 1), :]
    end
    NamedTuple{names}(comps)
end

"""
Foreach component whose batch dimension size is 1, pad it along the batch dimension 
to grow to the full batch size.
"""
function inflate_batch(batch::BatchTuple)
    bs = batch.batch_size
    @set batch.val = map(batch.val) do v
        if size(v)[end] == 1
            repeat(v, ntuple(Returns(1), ndims(v) - 1)..., bs)
        else
            @smart_assert size(v)[end] == bs
            v
        end
    end
end

function check_comp_size(
    batch::BatchTuple, comp_sizes::NamedNTuple{names,Int}
) where {names}
    Zygote.ignore() do
        @smart_assert keys(batch.val) == keys(comp_sizes)
        foreach(batch.val, comp_sizes) do v, dim
            @smart_assert size(v, 1) == dim
        end
    end
end

function check_approx(batch1::BatchTuple, batch2::BatchTuple; kargs...)
    @smart_assert keys(batch1.val) == keys(batch2.val)
    foreach(keys(batch1.val)) do k
        isapprox(batch1.val[k], batch2.val[k]; kargs...) || error(
            "Faild for componenet $k.\n\tLeft = $(batch1.val[k])\n\tRight = $(batch2.val[k])",
        )
    end
end

const should_check_finite = Ref(false)
function assert_finite(x::AbstractArray)
    if should_check_finite[] && !all(isfinite, x)
        Zygote.ignore() do
            @error "Some elements are not finite" x
            throw(ErrorException("assert_finite failed."))
        end
    end
    x
end
function assert_finite(x::BatchTuple)
    if should_check_finite[]
        Zygote.ignore() do
            for (k, v) in pairs(x.val)
                if !all(isfinite, v)
                    @error "In component $k, some elements are not finite" v
                    throw(ErrorException("assert_finite failed."))
                end
            end
        end
    end
    x
end

export plot_batched_series

"""
A general plotting function that plots a time series of `BatchedTuple`s. Each batch 
component will be plotted inside a separate subplot.

## Keyword args
- `mode`: :particle or :marginal. :particle plots a seperate line for each batch element, 
whereas :marginal plots a ribbon that shows the 20th, 50th, and 80th percentiles of 
each compoenent.

## Examples
```julia
let
    tconf = TensorConfig(false)
    times = 0:0.1:5
    series = [
        BatchTuple(tconf, 
            [
                (
                    pos=[randn(Float32), randn(Float32)+2], 
                    θ=[randn(Float32) * sin(Float32(t))],
                ) 
                for b in 1:3
            ]) 
        for t in times
    ]

    plot_batched_series(times, series)
end
```
"""
function plot_batched_series(
    times,
    series::TimeSeries{<:BatchTuple};
    mode=:marginal,
    truth::Union{Nothing,TimeSeries{<:BatchTuple}}=nothing,
    plot_width=600,
    plot_args...,
)
    times = TensorConfig(false).(times)
    series = TensorConfig(false).(series)
    truth === nothing || (truth = TensorConfig(false).(truth))
    plot_batched_series(Val(mode), times, series; truth, plot_width, plot_args...)
end

function plot_batched_series(
    ::Val{:particle},
    times,
    series::TimeSeries{<:BatchTuple};
    truth::Union{Nothing,TimeSeries{<:BatchTuple}}=nothing,
    plot_width=600,
    plot_args...,
)
    @smart_assert length(times) == length(series)
    to_plot_data(ys::TimeSeries{<:Real}) = (; xs=times, ys, alpha=1.0)
    to_plot_data(ys::TimeSeries{<:AbsMat{<:Real}}) = begin
        local (dim, batch_size) = size(ys[1])
        local Num = eltype(ys[1])
        local T = length(ys)
        local x_out = repeat([times; NaN], batch_size)
        local y_out = Matrix{Num}(undef, batch_size * (T + 1), dim)
        for i in 1:batch_size
            for t in 1:T
                y_out[(i - 1) * (T + 1) + t, :] = ys[t][:, i]
            end
            y_out[i * (T + 1), :] .= NaN
        end
        (; xs=x_out, ys=y_out, alpha=0.6 / sqrt(batch_size))
    end

    template = series[1]
    if truth !== nothing
        truth_temp = truth[1]
        @smart_assert truth_temp.batch_size == 1
        @smart_assert keys(truth_temp.val) == keys(template.val)
    end

    subplots = []
    for k in keys(template.val)
        xs, ys, linealpha = to_plot_data((b -> b.val[k]).(series))
        n_comps = size(template.val[k], 1)
        linecolor = hcatreduce(1:n_comps)
        label = ["$k[$i]" for i in 1:n_comps] |> hcatreduce
        push!(subplots, plot(xs, ys; linealpha, linecolor, label))
        if truth !== nothing
            txs, tys, _ = to_plot_data((b -> b.val[k]).(truth))
            plot!(
                txs, tys; label=hcatreduce(["$k[$i] (truth)" for i in 1:n_comps]), linecolor
            )
        end
    end
    plot(
        subplots...;
        layout=(length(subplots), 1),
        size=(plot_width, 0.6plot_width * length(subplots)),
        left_margin=1.5cm,
        plot_args...,
    )
end

function plot_batched_series(
    ::Val{:marginal},
    times,
    series::TimeSeries{<:BatchTuple};
    truth::Union{Nothing,TimeSeries{<:BatchTuple}}=nothing,
    plot_width=600,
    quantile_spread=0.25,
    component_names=nothing,
    plot_args...,
)
    @smart_assert 0 <= quantile_spread <= 0.5
    @smart_assert length(times) == length(series)

    plot_data!(ys::TimeSeries{<:Real}; args...) = plot!(times, ys, args...)
    plot_data!(ys::TimeSeries{<:AbsVec{<:Real}}; args...) = begin
        @unzip lower, middle, upper = map(ys) do ensemble
            local l, m, u = quantile(
                ensemble, (0.5 - quantile_spread, 0.5, 0.5 + quantile_spread)
            )
            m - l, m, u - m
        end
        plot!(times, middle; ribbon=(lower, upper), args...)
    end
    template = series[1]

    subplots = []
    for k in keys(template.val)
        n_dims = size(template.val[k], 1)
        sp = plot()
        for d in 1:n_dims
            local ys = (b -> b.val[k][d, :]).(series)
            comp_name = if component_names === nothing
                "$k[$d]"
            else
                component_names[k][d]
            end
            plot_data!(
                ys;
                linecolor=d,
                fillcolor=d,
                fillalpha=0.4,
                line=:dot,
                label="$comp_name (median)",
            )
            truth === nothing || plot_data!(
                (b -> b.val[k][d, :]).(truth); linecolor=d, label="$comp_name (truth)"
            )
        end
        push!(subplots, sp)
    end
    plot(
        subplots...;
        layout=(length(subplots), 1),
        size=(plot_width, 0.6plot_width * length(subplots)),
        left_margin=1.5cm,
        plot_args...,
    )
end
##-----------------------------------------------------------
export FluxLayer
"""
    FluxLayer(forward, Val(layer_name), trainable, [layer_info=trainable])

An flexible wrapper to represent an arbitrary trainable computation. This can be 
used to implement new forward computation logic without declaring a new struct.

- `forward::Function` should have the signature `forward(trainable)(args...) -> result`. i.e., 
it should be a higher-order function that first takes in the trainable parameters and then 
the input arguments of the layer. See the examples below.
- `trainable` can be either a Tuple or NamedTuple. 
- `layer_info=trainable` can be anything and will only be used to display the layer.

## Examples
```julia
julia> layer = FluxLayer(
           Val(:MyBilinear), 
           (w=rand(2,3),), 
           info=(x_dim = 3, y_dim = 2)
       ) do trainable
           (x, y) -> y' * trainable.w * x
       end
FluxLayer{MyBilinear}((x_dim = 3, y_dim = 2))

julia> layer(ones(3), ones(2))
2.7499240871541684

julia> Flux.params(layer)
Params([[0.6650776972679167 0.004201998819206465 0.7122138939453165; 
0.2132508860808494 0.17734661433873555 0.977832996702144]])
```
"""
struct FluxLayer{name,PS<:Union{Tuple,NamedTuple},F<:Function,Info}
    forward::F
    layer_name::Val{name}
    trainable::PS
    info::Info
end
Flux.@functor FluxLayer

FluxLayer(forward::Function, layer_name, trainable; info=trainable) =
    FluxLayer(forward, layer_name, trainable, info)

function Base.show(io::IO, ::Type{<:FluxLayer{name}}) where {name}
    print(io, "FluxLayer{$name, ...}")
end

function Base.show(io::IO, l::FluxLayer{name}) where {name}
    print(io, "FluxLayer{$name}($(l.info))")
end

function (l::FluxLayer)(args...)
    l.forward(l.trainable)(args...)
end

function Flux.trainable(l::FluxLayer)
    Flux.trainable(l.trainable)
end
##-----------------------------------------------------------
# trainable multivairate Normal distributions
"""
Currently, this is only supported on CPUs due to the underlying Distributions.jl's 
implementation.
"""
struct MvNormalLayer{V<:AbsVec,M<:AbsMat}
    μ::V
    U::M  # used as an upper-triangular matrix (the Cholesky decomposition of Σ)
end
Flux.@functor MvNormalLayer

function Statistics.cov(layer::MvNormalLayer)
    # generate a mask that is 1 for all upper-triangular elements
    U = UpperTriangular(layer.U)
    PDMat(Cholesky(U))
end

function rand(layer::MvNormalLayer, n::Integer)
    rand(MvNormal(layer.μ, cov(layer)), n)
end

function logpdf(layer::MvNormalLayer, x)
    logpdf(MvNormal(layer.μ, cov(layer)), x)
end
##-----------------------------------------------------------
"""
A wrapper over a trainalbe array parameter that carries the information about whether it 
requires regularization.
"""
struct Trainable{regular,M<:AbstractArray}
    is_regular::Val{regular}
    array::M
end
Flux.@functor Trainable
Trainable(regular::Bool, a::AbstractArray) = Trainable(Val(regular), a)

Base.show(io::IO, x::Trainable{regular}) where {regular} = begin
    print(io, "Trainable(regular=$(regular), $(summary(x.array)))")
end

"""
Return an iterator of trainable parameters that require regularization.
You can also wrap an array `x` inside a [`Trainale`](@ref), which would allow you 
to specify whether `x` needs regularization.
"""
function regular_params end

regular_params(layer::Trainable{true}) = (layer.array,)
regular_params(layer::Trainable{false}) = tuple()


regular_params(layer::Union{FluxLayer,Chain,SkipConnection}) =
    Iterators.flatten(map(regular_params, Flux.trainable(layer)))

regular_params(layer::Dense) = (layer.weight,)

regular_params(layer::Union{Flux.GRUCell,Flux.LSTMCell}) = (layer.Wi, layer.Wh)

regular_params(tp::Union{Tuple,NamedTuple}) = Iterators.flatten(map(regular_params, tp))

regular_params(::Union{Function,BatchNorm,Nothing}) = tuple()

# fix Flux Params
Base.in(k::Flux.Params, v::Base.KeySet{Any,<:IdDict}) =
    get(v.dict, k, Base.secret_table_token) !== Base.secret_table_token