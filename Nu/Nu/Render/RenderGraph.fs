namespace RenderGraph   

open System
open System.Collections.Generic
open System.Numerics
open Prime
open Nu

// RenderGraph is our way to describe rendering in a backend-agnostic way—like an API that works with OpenGL, Vulkan, WebGL, whatever. 
// It lets us define what we want to render (resources, actions, dependencies) without tying to specific graphics APIs. 
// This helps support multiple backends easily: we build the graph once, and each backend (like our OpenGL executor) interprets it differently. 
// Plus, it enables cool features like automatic optimization (batching, dead code elimination) and makes complex renders (deferred lighting, post-effects) modular and reusable. 
// Think of it as a blueprint: resources are the materials/tools, actions are the steps, edges show dependencies to avoid conflicts.

//RenderGraph should be backend agnostic almost like an API
//We try to define some universal abstractions that end-user will work with while comromising with our backends OpenGL, Vulcan, Webgl
//This is all just suggestion of how such type-declarations might look like. It may be correct or not

// ColorFormat defines how colors are stored in textures or buffers, covering different bit depths and channel combinations (R for red, G for green, B for blue, A for alpha/transparency).
// We need this to balance quality, memory, and performance—for example, RGBA8 is a cheap, common choice for basic sprites, while RGBA32F gives high precision for effects like HDR lighting, which helps avoid banding or loss of detail in bright/dark areas.
type ColorFormat = 
    | RGBA8 | RGBA16F | RGBA32F
    | RGB8 | RGB16F | RGB32F  
    | RG8 | RG16F | RG32F
    | R8 | R16F | R32F
    | R11G11B10F
    | RGB10A2

// DepthFormat handles how depth information is stored for 3D rendering, basically tracking how far away things are from the camera with options like 16-bit (D16) or floating-point 32-bit (D32F).
// It's crucial for depth testing to prevent drawing objects behind others, and higher precision formats reduce issues like z-fighting (flickering when surfaces are too close), making it essential for clean 3D scenes or shadow mapping.
type DepthFormat = D16 | D24 | D32F

// StencilFormat is for stencil buffers, which act like a mask to control where rendering happens—usually just S8 for 8 bits of masking data.
// This is handy for advanced effects like drawing only inside a portal or clipping UI elements, and we include it because it allows selective rendering without extra passes, saving performance in complex setups like reflections or outlines.
type StencilFormat = S8

/// Combined depth-stencil formats pack both depth and stencil into one texture, like D24S8 (24-bit depth + 8-bit stencil).
// We use this to save memory and bandwidth when a pass needs both, which is common in deferred rendering or shadows with masking, making the graph more efficient for real-world scenes.
type DepthStencilFormat = D24S8 | D32FS8

// PixelFormat wraps up all the possible formats for textures or buffers, whether it's color, depth, stencil, or a combo.
// This gives us a single way to specify what a texture holds, ensuring backends create the right resource type—it's key for consistency across different graphics APIs.
type PixelFormat =
    | Color of ColorFormat
    | Depth of DepthFormat
    | Stencil of StencilFormat
    | DepthStencil of DepthStencilFormat

// GeometryType describes the shape or primitive we're drawing, from simple fullscreen triangles for post-effects to quads for sprites, meshes via ID for 3D models, or even procedural generators.
// Abstracting this lets backends handle the vertex data internally, which is great for portability—plus, it supports optimizations like switching to instanced drawing for batches of similar geometry.
type GeometryType =
    | FullscreenTriangle
    | Quad
    | Mesh of meshId: Guid
    | Procedural of generator: string

// BlendFactor controls the factors in blending equations, like using source alpha (SrcAlpha) or its inverse (OneMinusSrcAlpha) to mix colors.
// This is what makes transparency work smoothly, and we need separate factors for flexibility— for instance, it allows effects like additive blending for glows, which adds colors without darkening, perfect for particles or lights.
type BlendFactor = 
    | Zero | One 
    | SrcColor | OneMinusSrcColor
    | DstColor | OneMinusDstColor
    | SrcAlpha | OneMinusSrcAlpha
    | DstAlpha | OneMinusDstAlpha

// BlendOp defines the math operation for blending, such as adding or subtracting colors.
// Having ops like Add or Min lets us customize behaviors beyond basic transparency—Add is ideal for brightening effects, and separating it from factors gives more control, which is why we can create varied visuals like screen overlays or darken filters efficiently.
type BlendOp = Add | Subtract | ReverseSubtract | Min | Max

// BlendState combines blend factors and ops for both color and alpha channels.
// It allows per-material blending without relying on global states, which prevents bugs in complex graphs and makes it easier to mix effects like opaque objects with translucent ones in the same frame.
type BlendState = {
    ColorSrc: BlendFactor
    ColorDst: BlendFactor
    ColorOp: BlendOp
    AlphaSrc: BlendFactor
    AlphaDst: BlendFactor
    AlphaOp: BlendOp
}

// BlendStates offers quick presets like Opaque (no blending), AlphaBlend for transparency, or Additive for overlays.
// These defaults speed up setup for common cases, and since blending can be performance-heavy, having them encourages efficient use—like using Additive for fire effects to accumulate brightness without extra passes.
module BlendStates =
    let Opaque = None
    let AlphaBlend = Some { 
        ColorSrc = SrcAlpha; ColorDst = OneMinusSrcAlpha; ColorOp = Add
        AlphaSrc = One; AlphaDst = OneMinusSrcAlpha; AlphaOp = Add 
    }
    let Additive = Some {
        ColorSrc = SrcAlpha; ColorDst = One; ColorOp = Add
        AlphaSrc = One; AlphaDst = One; AlphaOp = Add
    }

//We expect different shadertechniques to have different rendering approach so we need to come up with some boundaries as early as possible
//for example Sprite/2d is for sprites, deferred would use 6-buffer shaders, fullscreenpost has no vertex transforms, compute etc...
// ShaderTechnique categorizes shaders by their main purpose, like Sprite for 2D rendering or Deferred for advanced lighting.
// Grouping like this helps backends pick the right compilation path or optimizations— for example, Sprite might assume orthographic projection, which simplifies 2D games, and it keeps the graph organized for mixing techniques in one scene.
type ShaderTechnique = 
    | Sprite //for 2d renders
    | FullscreenPost  
    | Deferred
    | Compute
    | Custom of string

/// ShaderFeatures are sets of tags like "Instanced" or "Skinned" to create shader variants.
// This lets us compile specialized versions on the fly, avoiding a explosion of shader files—it's especially useful for performance, as features like instancing reduce draw calls in batches, and it makes the system adaptable to different hardware.
type ShaderFeatures = Set<string> // "Instanced", "AlphaTest", "Skinned"

/// ResourceHandle is a strongly-typed handle for resources, with a Guid ID, version number, and the resource type itself.
// The ID and version track uniqueness and updates (like hot-reloading), which is vital for caching and avoiding stale data in long-running apps or during development.
type ResourceHandle<'T> = { Id: Guid; Version: int; ResourceType: 'T }

/// UniformValue represents shader parameters, either static constants (like floats or matrices) or dynamic functions that pull from the world state.
// Separating static and dynamic helps optimize—statics can be set once, while dynamics update per frame, and this abstraction keeps the graph engine-agnostic by avoiding direct world dependencies.
type UniformValue =
    | StaticFloat of single
    | StaticVec2 of Vector2
    | StaticVec3 of Vector3
    | StaticVec4 of Vector4
    | StaticColor of Color
    | StaticMatrix of Matrix4x4
    | StaticTransform of Transform
    //obj would be world but may not be smart to introduce dependency on world here? We want to keep RenderGraph oblivious to game engine? 
    | DynamicFloat of (obj -> single)
    | DynamicVec2 of (obj -> Vector2)
    | DynamicVec3 of (obj -> Vector3)
    | DynamicVec4 of (obj -> Vector4)
    | DynamicColor of (obj -> Color)
    | DynamicMatrix of (obj -> Matrix4x4)

// TextureResourceType specifies a texture's properties, including width, height, pixel format, and sample count for multisampling (MSAA).
// This detail level ensures backends allocate correctly, and features like MSAA smooth edges without much extra code, which is why it's great for high-quality renders on varying hardware.
type TextureResourceType = { 
    Width: int
    Height: int
    Format: PixelFormat
    Samples: int  // For MSAA
}

// BufferResourceType describes non-texture buffers, with size and usage hint (like "Vertex" or "Uniform").
// It abstracts things like vertex buffers, allowing the graph to handle data storage generically, which simplifies porting to backends with different buffer types.
type BufferResourceType = { Size: int; Usage: string }

// UniformResourceType is for uniform blocks, holding size and the data object.
// Larger uniform sets benefit from this, as it binds chunks of data at once, reducing overhead in shaders with many parameters.
type UniformResourceType = { Size: int; UniformData: obj }

/// ShaderResourceType defines a shader via its technique, features, and a cache key generated from them.
// Caching by key prevents recompiling the same shader repeatedly, saving time especially in variant-heavy systems, and it ties directly into how backends load or generate code.
type ShaderResourceType = {
    Technique: ShaderTechnique
    Features: ShaderFeatures
    CacheKey: string  // Auto-generated from technique + features
}

/// RenderState captures drawing states like blending, depth testing, and culling.
// Per-material states like this avoid global changes that could break other parts of the graph, making it safer for concurrent or multi-pass rendering.
type RenderState = {
    Blending: BlendState option
    DepthTest: bool option  
    CullMode: string option  // "None", "Front", "Back"
}

/// MaterialInstance pairs a shader with specific uniforms, textures, render state, and an optional cache key.
// This lets multiple objects share a shader but customize params, which is efficient for memory and draw calls—caching identical instances further optimizes by reusing setups.
type MaterialInstance = {
    ShaderId: Guid  // Reference to shared shader resource
    Uniforms: Map<string, UniformValue>
    Textures: Map<string, Image AssetTag>
    RenderState: RenderState
    CacheKey: string option  // For sharing identical materials
}

/// MaterialResourceType wraps a material instance and a flag for batch compatibility.
// The batch flag enables grouping similar materials for instanced drawing, slashing CPU overhead in scenes with many similar objects like sprites or particles.
type MaterialResourceType = {
    MaterialInstance: MaterialInstance
    BatchCompatible: bool  // Can this material be batched with others?
}

/// InstanceData holds per-instance info for GPU batching, like transform, color, UV offsets, and custom vectors.
// Sending this to the GPU in one go avoids separate draw calls, boosting perf massively for crowds or UI elements, which is why instancing is a game-changer for scalability.
type InstanceData = {
    Transform: Matrix4x4
    Color: Color
    UV: Vector4  // For texture atlas/animation
    CustomData: Vector4  // Material-specific instance data
}

// InstanceBufferType manages arrays of instance data, with a max count and usage hint.
// It's the buffer behind instancing, allowing dynamic updates for moving objects, and keeps batches efficient even in animated scenes.
type InstanceBufferType = {
    InstanceData: InstanceData array
    MaxInstances: int
    Usage: string
}

/// ResourceLifetime manages how long resources last—transient for one frame, persistent until destroyed, or imported from outside.
// This controls memory usage smartly: transients auto-clean for temp buffers, while persistents cache expensive things like shaders, preventing leaks or reloads.
type ResourceLifetime =
    | Transient  // Lives for one frame
    | Persistent // Lives until explicitly destroyed
    | Imported   // Managed externally (e.g., existing Nu assets)




/// ActionLifetime controls how long or when an action runs, like "do this for 3 frames" or "start after 2 frames."
// This is key for temporal effects like motion blur, where you need data across multiple frames, or for delayed actions like a fade-in.
// It makes the graph flexible for animations or multi-frame processes without hardcoding frame counts in the executor.
type ActionLifetime =
    | Immediate  // Runs this frame only
    | FixedFrames of int  // Runs for n frames
    | Delayed of startFrame: int * duration: int  // Starts after startFrame, runs for duration


/// ResourceNode declares what resources the graph needs, from textures and buffers to shaders and materials.
// These are the building blocks added to the graph, ensuring all dependencies are explicit for automatic ordering and validation.
type ResourceNode =
    | TextureResource of TextureResourceType * ResourceLifetime
    | BufferResource of BufferResourceType * ResourceLifetime  
    | UniformResource of UniformResourceType * ResourceLifetime
    | ShaderResource of ShaderResourceType * ResourceLifetime
    | MaterialResource of MaterialResourceType * ResourceLifetime
    | InstanceBufferResource of InstanceBufferType * ResourceLifetime
    | ImportedTextureResource of Image AssetTag

/// ResourceAccess specifies how a resource is used in an action—read-only for sampling, write-only for rendering to, or read-write.
// This hints at potential conflicts or optimizations, like barriers in Vulkan, making the graph safer and more performant across backends.
type ResourceAccess =
    | ReadOnly
    | WriteOnly
    | ReadWrite

/// ResourceBinding ties a resource handle to its access mode and a name for reference.
// It's how actions link to resources, providing context for bindings in shaders or framebuffers.
type ResourceBinding<'T> = {
    Handle: ResourceHandle<'T>
    Access: ResourceAccess
    Name: string
}


//---------------------------------------------------------------------------- Action Types  


/// ShaderActionDescriptor outlines a shader-based draw call, including material, output targets (with MRT for multiple), geometry, and instancing options.
// This is the core of rendering steps, supporting things like drawing to multiple textures at once for deferred effects, and instancing reduces calls for better FPS in busy scenes.
type ShaderActionDescriptor = {
    Material: ResourceBinding<MaterialResourceType>
    ColorOutputs: (int * ResourceBinding<TextureResourceType>) list
    DepthOutput: ResourceBinding<TextureResourceType> option
    StencilOutput: ResourceBinding<TextureResourceType> option
    Geometry: GeometryType
    InstanceBuffer: ResourceBinding<InstanceBufferType> option
    InstanceCount: int
    Lifetime: ActionLifetime 
}


/// ImmediateActionDescriptor for one-off rendering that happens right now, no fuss.
// We need this for quick draws like debug lines, UI overlays, or temp effects that don't need to hook into the full dependency graph.
// It's super lightweight—specify a shader, geometry, and basic params, and it renders in the current frame, bypassing heavy scheduling.
type ImmediateActionDescriptor = {
    Shader: ResourceBinding<ShaderResourceType>
    Geometry: GeometryType
    Uniforms: Map<string, UniformValue>
    Textures: Map<string, Image AssetTag>
    RenderState: RenderState
}


//I have no idea what im doing here :) Just stubs for now compute shaders seem important though
type ComputeActionDescriptor = {
    Shader: ResourceBinding<ShaderResourceType>
    Inputs: ResourceBinding<BufferResourceType> list
    Outputs: ResourceBinding<BufferResourceType> list
    DispatchSize: Vector3i
    Lifetime: ActionLifetime
}

// ClearActionDescriptor is for clearing a target texture to a specific color.
// It's a simple way to reset buffers before drawing, essential for avoiding garbage from previous frames in multi-pass setups.
type ClearActionDescriptor = {
    Target: ResourceBinding<TextureResourceType>
    Color: Color
}

// PresentActionDescriptor marks the final step to display a source texture on screen.
// This signals the end of the graph, handling things like swapping to the backbuffer, and keeps presentation separate for flexibility in windowed or fullscreen modes.
type PresentActionDescriptor = {
    Source: ResourceBinding<TextureResourceType>
}

/// ActionNode represents operations in the graph, like shader draws, computes, clears, or presents.
// These are the active steps that use resources, with dependencies ensuring correct order—no reading before writing, for example.
type ActionNode =
    | ShaderAction of ShaderActionDescriptor
    | ComputeAction of ComputeActionDescriptor
    | ClearAction of ClearActionDescriptor
    | PresentAction of PresentActionDescriptor
    | ImmediateAction of ImmediateActionDescriptor



// ------------------------------------------------------------------- Render Pass Support (Stub for now)

// LoadOp and StoreOp control how attachments are handled at the start and end of a pass, like clearing on load or resolving MSAA on store.
// These ops optimize by skipping unnecessary work—DontCare avoids loading old data, which saves bandwidth in tight loops like post-processing chains.
type LoadOp = Clear of Color | Load | DontCare
type StoreOp = Store | DontCare | Resolve  // Resolve for MSAA

// RenderPassDescriptor groups actions into a single pass with attachments and their ops.
// It's a stub for now but will allow binding multiple targets at once, like in framebuffers, which is efficient for passes needing color + depth together.
type RenderPassDescriptor = {
    Name: string
    ColorAttachments: (int * Guid * LoadOp * StoreOp) list
    DepthAttachment: (Guid * LoadOp * StoreOp) option
    Actions: Guid list  // Actions in this pass
}


// -------------------------------------------------------------------------------- Graph Structure


/// RenderGraph is the main structure holding nodes, edges, resources, actions, the output action, and batching data.
// It's the complete description we build and pass to executors, with built-in support for optimizations like batching compatible actions.
type RenderGraph = {
    Nodes: Map<Guid, RenderNode>
    Edges: (Guid * Guid) list
    Resources: Map<Guid, ResourceNode>
    Actions: Map<Guid, ActionNode>
    OutputAction: Guid
    Name: string
    // GPU instancing optimization data
    BatchableActions: Map<Guid, Guid list>  // Shader ID -> List of compatible actions
}

/// RenderNode combines resources, actions, passes, or even sub-graphs into one type.
// This hierarchy allows nesting graphs for reuse, like a post-effect sub-graph plugged into multiple mains.
and RenderNode =
    | Resource of Guid * ResourceNode
    | Action of Guid * ActionNode
    | Pass of Guid * RenderPassDescriptor  
    | SubGraph of Guid * RenderGraph


// -------------------------------------------------------------------------- Helper Functions, builder helpers further below
//These helpers dont have to live here or user can write their own RenderGraph functions ideally we want to have some abstractions
//for common things that graph can do like render a 2d sprite but also let users write their own graphs/sub-graphs which will ALWAYS work on every backend

module RenderGraph =
    
    // Create typed resource handle
    let createResourceHandle : 'T -> ResourceHandle<'T> = 
        fun resourceType -> { 
            Id = Guid.NewGuid()
            Version = 0
            ResourceType = resourceType
        }
    
    /// Increment resource version—for tracking changes without recreating everything.
    let incrementVersion handle = { handle with Version = handle.Version + 1 }
    
    // Create empty render graph
    let empty name = {
        Nodes = Map.empty
        Edges = []
        Resources = Map.empty
        Actions = Map.empty
        OutputAction = Guid.Empty
        Name = name
        BatchableActions = Map.empty
    }
    
    // Add a resource node
    let addResource resourceId resourceNode graph =
        let node = Resource (resourceId, resourceNode)
        { graph with 
            Nodes = Map.add resourceId node graph.Nodes
            Resources = Map.add resourceId resourceNode graph.Resources }
    
    // Add an action node
    let addAction actionId actionNode graph =
        let node = Action (actionId, actionNode)
        { graph with 
            Nodes = Map.add actionId node graph.Nodes
            Actions = Map.add actionId actionNode graph.Actions }
    
    // Add a sub-graph node
    let addSubGraph subGraphId subGraph graph =
        let node = SubGraph (subGraphId, subGraph)
        { graph with Nodes = Map.add subGraphId node graph.Nodes }
    
    /// Add dependency edge—to enforce order between nodes.
    let addDependency sourceId targetId graph =
        { graph with Edges = (sourceId, targetId) :: graph.Edges }
    
    // Set output action
    let setOutputAction actionId graph =
        { graph with OutputAction = actionId }
    
    // Get resource dependencies for an action
    let getResourceDependencies action =
        match action with
        | ShaderAction desc -> 
            let materialDep = desc.Material.Handle.Id
            let colorDeps = desc.ColorOutputs |> List.map (fun (_, binding) -> binding.Handle.Id)
            let depthDeps = desc.DepthOutput |> Option.map (fun b -> b.Handle.Id) |> Option.toList
            let instanceDeps = desc.InstanceBuffer |> Option.map (fun b -> b.Handle.Id) |> Option.toList
            materialDep :: (colorDeps @ depthDeps @ instanceDeps)
        | ComputeAction desc ->
            let shaderDep = desc.Shader.Handle.Id
            let inputDeps = desc.Inputs |> List.map (fun b -> b.Handle.Id)
            let outputDeps = desc.Outputs |> List.map (fun b -> b.Handle.Id)
            shaderDep :: (inputDeps @ outputDeps)
        | ClearAction desc -> [desc.Target.Handle.Id]
        | PresentAction desc -> [desc.Source.Handle.Id]
        | ImmediateAction desc -> [desc.Shader.Handle.Id]
    
    // Get shader ID from a material resource
    let getShaderIdFromMaterial materialId graph =
        match Map.tryFind materialId graph.Resources with
        | Some (MaterialResource (materialType, _)) -> 
            Some materialType.MaterialInstance.ShaderId
        | _ -> None
    
    // Automatically build dependency edges based on resource usage
    let buildDependencyEdges graph =
        let edges = ResizeArray<Guid * Guid>()
        
        // Add resource-to-resource dependencies
        for KeyValue(resourceId, resource) in graph.Resources do
            match resource with
            | MaterialResource (materialType, _) ->
                // Material depends on its shader
                edges.Add((materialType.MaterialInstance.ShaderId, resourceId))
            | _ -> ()
        
        // Add resource-to-action dependencies
        for KeyValue(actionId, action) in graph.Actions do
            let resourceDeps = getResourceDependencies action
            for resourceId in resourceDeps do
                edges.Add((resourceId, actionId))
        
        { graph with Edges = edges |> List.ofSeq }
    
    // Validate graph for resource access conflicts
    let validateGraph graph =
        for (sourceId, targetId) in graph.Edges do
            match Map.tryFind targetId graph.Actions with
            | Some action ->
                let resourceDeps = getResourceDependencies action
                for resourceId in resourceDeps do
                    if not (Map.containsKey resourceId graph.Resources) then
                        failwith $"Action {targetId} depends on non-existent resource {resourceId}"
            | None -> ()
        graph
    
    // Topological sort
    let topologicalSort graph =
        let rec sort visited sorted remaining depth =
            if depth > 100 then failwith "Max recursion depth exceeded in topological sort"
            
            match remaining with
            | [] -> List.rev sorted
            | _ ->
                let ready = 
                    remaining 
                    |> List.filter (fun nodeId -> 
                        graph.Edges
                        |> List.filter (fun (_, targetId) -> targetId = nodeId)
                        |> List.map fst
                        |> List.forall (fun depId -> List.contains depId visited))
                
                match ready with
                | [] -> 
                    if List.isEmpty remaining then List.rev sorted
                    else failwith "Circular dependency detected in render graph"
                | readyNodes ->
                    let newVisited = readyNodes @ visited
                    let newSorted = readyNodes @ sorted
                    let newRemaining = remaining |> List.filter (fun id -> not (List.contains id readyNodes))
                    sort newVisited newSorted newRemaining (depth + 1)
        
        let allNodes = graph.Nodes |> Map.toList |> List.map fst
        sort [] [] allNodes 0

    // Dead code elimination - prune unreachable nodes
    let pruneUnusedNodes graph =
        let rec collectReachable nodeId visited =
            if Set.contains nodeId visited then visited
            else
                let visited' = Set.add nodeId visited
                let dependencies = 
                    graph.Edges 
                    |> List.filter (fun (_, target) -> target = nodeId)
                    |> List.map fst
                
                List.fold (fun acc dep -> collectReachable dep acc) visited' dependencies
        
        let reachable = collectReachable graph.OutputAction Set.empty
        
        { graph with
            Nodes = graph.Nodes |> Map.filter (fun id _ -> Set.contains id reachable)
            Resources = graph.Resources |> Map.filter (fun id _ -> Set.contains id reachable)
            Actions = graph.Actions |> Map.filter (fun id _ -> Set.contains id reachable)
            Edges = graph.Edges |> List.filter (fun (src, dst) -> 
                Set.contains src reachable && Set.contains dst reachable) }


//  ------------------------------------------------------------------------------Builder Helpers

module RenderGraphBuilder =
    
    // Create a shared shader resource (persistent for reuse across materials)
    let createShaderResource technique features =
        let cacheKey = $"""{technique}_{features |> Set.toList |> List.map string |> String.concat "_"}"""
        let shaderResourceType = { Technique = technique; Features = features; CacheKey = cacheKey }
        let shaderId = Guid.NewGuid()
        (shaderId, ShaderResource (shaderResourceType, Persistent))
    
    // Create a material instance
    let createMaterialInstance shaderId uniforms textures renderState cacheKey =
        {
            ShaderId = shaderId
            Uniforms = uniforms
            Textures = textures
            RenderState = renderState
            CacheKey = cacheKey
        }
    
    // Create a material resource
    let createMaterialResource materialInstance batchCompatible =
        let materialResourceType = { 
            MaterialInstance = materialInstance
            BatchCompatible = batchCompatible
        }
        let materialId = Guid.NewGuid()
        (materialId, MaterialResource (materialResourceType, Transient), materialResourceType)
    
    // Create instance buffer for GPU instancing
    let createInstanceBuffer instanceData maxInstances =
        let instanceBufferType = {
            InstanceData = instanceData
            MaxInstances = maxInstances
            Usage = "Dynamic"
        }
        let bufferId = Guid.NewGuid()
        (bufferId, InstanceBufferResource (instanceBufferType, Transient), instanceBufferType)
    
    // Create a sprite graph with backwards compatibility
    let createSpriteGraph name vertexSrc fragmentSrc uniforms textures blendMode depthTest =
        let graph = RenderGraph.empty name
        
        // Create shader resource with Sprite technique
        let (shaderResourceId, shaderResource) = createShaderResource Sprite Set.empty
        
        // Convert old blend mode to new render state
        let renderState = {
            Blending = 
                match blendMode with
                | Some "Transparent" -> BlendStates.AlphaBlend
                | Some "Additive" -> BlendStates.Additive
                | Some "Overwrite" -> BlendStates.Opaque
                | _ -> BlendStates.AlphaBlend
            DepthTest = depthTest
            CullMode = None
        }
        
        // Create material instance
        let materialInstance = createMaterialInstance shaderResourceId uniforms textures renderState None
        let (materialId, materialResource, materialResourceType) = createMaterialResource materialInstance true
        
        // Create output texture resource
        let outputTextureType = { Width = 0; Height = 0; Format = Color RGBA8; Samples = 1 }
        let outputId = Guid.NewGuid()
        let outputResource = TextureResource (outputTextureType, Transient)
        
        let actionId = Guid.NewGuid()
        
        let shaderAction = ShaderAction {
            Material = {
                Handle = { Id = materialId; Version = 0; ResourceType = materialResourceType }
                Access = ReadOnly
                Name = "material"
            }
            ColorOutputs = [(0, {
                Handle = { Id = outputId; Version = 0; ResourceType = outputTextureType }
                Access = WriteOnly
                Name = "output"
            })]
            DepthOutput = None
            StencilOutput = None
            Geometry = Quad
            InstanceBuffer = None
            InstanceCount = 1
            Lifetime = Immediate 
        }
        
        graph
        |> RenderGraph.addResource shaderResourceId shaderResource
        |> RenderGraph.addResource materialId materialResource
        |> RenderGraph.addResource outputId outputResource
        |> RenderGraph.addAction actionId shaderAction
        |> RenderGraph.setOutputAction actionId
        |> RenderGraph.buildDependencyEdges
        |> RenderGraph.validateGraph