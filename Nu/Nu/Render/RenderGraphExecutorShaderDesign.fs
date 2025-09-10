//shader module design exploration - will be integrated into main system, doesnt really belong in a separate file
namespace Nu

open System
open System.Numerics
open System.Collections.Concurrent
open Prime
open RenderGraph

//for for different loading strategies
type ShaderSource =
    | SingleGlslFile of string                         // Single file with #shader vertex/#shader fragment sections
    | SeparateFiles of vertex:string * fragment:string // Separate .vert and .frag files
    | InlineSource of vertex:string * fragment:string  // Direct source strings
    // Future: | ShaderGraph of nodes (or shader graph could give you inline source)

/// Base class for shader modules with automatic uniform extraction
[<AbstractClass>]
type Shader<'Props when 'Props : struct>() =
    
    static let propFields = Microsoft.FSharp.Reflection.FSharpType.GetRecordFields(typeof<'Props>)
    
    static let uniformExtractors = 
        propFields |> Array.map (fun field ->
            let getter = Microsoft.FSharp.Reflection.FSharpValue.PreComputeRecordFieldReader field
            let name = field.Name
            fun (props: obj) ->
                let value = getter props
                let uniform = 
                    match value with
                    | :? Color as c -> StaticColor c
                    | :? single as f -> StaticFloat f
                    | :? Vector2 as v -> StaticVec2 v
                    | :? Vector3 as v -> StaticVec3 v
                    | :? Vector4 as v -> StaticVec4 v
                    | :? Matrix4x4 as m -> StaticMatrix m
                    | :? Transform as t -> StaticTransform t
                    | _ -> failwithf "Unsupported uniform type: %A" (value.GetType())
                (name, uniform))
    
    abstract member Source: ShaderSource
    
    member _.ExtractUniforms(props: 'Props) =
        let boxed = box props
        uniformExtractors |> Array.map (fun f -> f boxed)


//caching ID of shaders - this is for batching in the graph executor
//identical shader graphs can be batched together
//furthermore, elsewhere identical uniforms shaders can be instanced automatically by the graph
module ShaderRegistry =
    // Each shader type gets a unique integer ID at runtime
    let mutable private nextId = 0
    let private idCache = ConcurrentDictionary<System.Type, int>()
    
    /// Get or create a unique ID for a shader type
    let getOrCreateId<'T>() =
        idCache.GetOrAdd(typeof<'T>, fun _ -> 
            System.Threading.Interlocked.Increment(&nextId))


//Render node with all data needed for batching decisions - WIP
type ShaderRenderNode = {
    ShaderId: int                           // Fast shader type comparison
    CacheKey: System.RuntimeTypeHandle      // For shader compilation cache (so we dont get lag like Godot everytime new shader shows up on screen)
    Properties: obj                          // Struct properties (boxed for storage)
    PropertiesHash: int                      // Pre-computed hash for fast comparison
    Uniforms: (string * UniformValue) array // Extracted uniforms
    Transform: Transform                     // Entity transform - things like transform or geometry likely belong in attributes category - can still vary accross instances
    ShaderSource: ShaderSource              // How to load shader
}

/// Data for instanced drawing
type InstancedDrawData = {
    ShaderId: int
    CacheKey: System.RuntimeTypeHandle
    Properties: obj  
    SharedUniforms: (string * UniformValue) array
    Transforms: Transform array
    ShaderSource: ShaderSource
}

/// Data for batched draws
type BatchedDrawsData = {
    ShaderId: int
    Nodes: ShaderRenderNode array
}

/// Render action after batching analysis
type BatchedRenderAction =
    | IndividualDraw of ShaderRenderNode
    | InstancedDraw of InstancedDrawData
    | BatchedDraws of BatchedDrawsData


type RenderGraphConfig = {
    EnableInstancing: bool     // Toggle instancing on/off (default: true)
    MinInstanceCount: int      // Min instances to trigger batching (default: 2)
    EnableStateSorting: bool   // Sort by shader to minimize state changes (default: true)
}

module RenderGraphConfig =
    let defaultConfig = {
        EnableInstancing = true
        MinInstanceCount = 2
        EnableStateSorting = true
    }


module RenderGraphOptimizer =
    
    /// Detect batching opportunities from render nodes
    let detectBatching (nodes: ShaderRenderNode list) (config: RenderGraphConfig) =
        if not config.EnableInstancing then
            // Instancing disabled - return individual actions
            nodes |> List.map IndividualDraw
        else
            // Group by shader type first (fast integer comparison)
            nodes
            |> List.groupBy (fun node -> node.ShaderId)
            |> List.collect (fun (shaderId, shaderNodes) ->
                if shaderNodes.Length < config.MinInstanceCount then
                    // Too few for batching
                    shaderNodes |> List.map IndividualDraw
                else
                    // Further group by properties for instancing
                    shaderNodes
                    |> List.groupBy (fun node -> node.PropertiesHash)
                    |> List.collect (fun (propsHash, samePropsNodes) ->
                        // Verify actual equality (hash collision check)
                        let groups = 
                            samePropsNodes 
                            |> List.groupBy (fun n -> n.Properties)
                        
                        groups |> List.map (fun (props, nodes) ->
                            if nodes.Length >= config.MinInstanceCount then
                                // INSTANCED: Same shader + same properties
                                InstancedDraw {
                                    ShaderId = shaderId
                                    CacheKey = nodes.Head.CacheKey
                                    Properties = props
                                    SharedUniforms = nodes.Head.Uniforms
                                    Transforms = nodes |> List.map (fun n -> n.Transform) |> Array.ofList
                                    ShaderSource = nodes.Head.ShaderSource
                                }
                            else
                                // BATCHED: Same shader, different properties (state sorting)
                                BatchedDraws {
                                    ShaderId = shaderId
                                    Nodes = nodes |> Array.ofList
                                })))
            |> if config.EnableStateSorting then
                   List.sortBy (function 
                       | IndividualDraw n -> n.ShaderId
                       | InstancedDraw i -> i.ShaderId
                       | BatchedDraws b -> b.ShaderId)
               else id

