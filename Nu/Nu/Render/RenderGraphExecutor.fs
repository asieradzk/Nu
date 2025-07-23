namespace Nu

open System
open System.Collections.Generic
open System.IO
open System.Numerics
open Prime
open RenderGraph

// This file is the engine that takes a RenderGraph and makes it real using OpenGL.
// In the future we want to have per-backend executr, this one would become RenderGraphExecutorOpenGl.fs 
// It’s like a translator, turning our abstract graph of resources, actions, and dependencies into actual GPU commands for drawing sprites, running shaders, or handling effects.
// We need this to keep our RenderGraph backend-agnostic, so the same graph can work with OpenGL today and potentially Vulkan or WebGL tomorrow, without changing the graph itself.
// It’s designed to be efficient—caching shaders, reusing geometry, and cleaning up properly—while staying flexible for complex rendering tasks.

// Runtime OpenGL resource handles—our way to track GPU objects like textures or shaders.
// Each variant holds an OpenGL ID and relevant data, so we can reference and manage them during rendering.
type RuntimeResource =
    | RuntimeTexture of uint32 * int * int
    | RuntimeShader of uint32 * Map<string, int> 
    | RuntimeMaterial of uint32 * Map<string, UniformValue> * Map<string, Image AssetTag>

// Shader source info—paths to vertex and fragment shader files.
// This links our abstract shader techniques (like Sprite) to actual GLSL files for compilation.
type ShaderSource = {
    VertexPath: string
    FragmentPath: string
}

// Executor's resource cache—stores shaders and materials for reuse across frames.
// This prevents us from recompiling shaders or recreating materials every frame, which would tank performance in a game with lots of objects.
type ExecutorResourceCache = {
    Shaders: Map<string, RuntimeResource>
    Materials: Map<string, RuntimeResource>
    ShaderSources: Map<ShaderTechnique * Set<string>, ShaderSource>  // Technique+Features -> paths
}

/// Executor state—everything we need to keep track of while rendering.
// It holds the cache for persistent resources, a map for frame-specific resources, and buffers for quad geometry, making sure we can render efficiently and clean up when done.
type ExecutorState = {
    Cache: ExecutorResourceCache
    FrameResources: Map<Guid, RuntimeResource>
    QuadVao: uint32 option
    QuadVbo: uint32 option
    QuadEbo: uint32 option
}

/// Asset resolver function type—converts Nu asset tags to actual textures.
// This connects our graph to Nu’s asset system, letting us pull in textures for rendering without hardcoding them.
type AssetResolver = Image AssetTag -> RenderAsset voption

module RenderGraphExecutor =
    
    //TODO: Not flexible change to a different/dynamic technique ASAP
    // Sets up the mapping from shader techniques (like Sprite) to their GLSL file paths.
    // This is super important because it tells the executor where to find the shader code for each technique, including variants like instanced rendering.
    // We do this once at startup to avoid looking up files every frame, keeping things fast.
    
    let private initializeShaderSources () =
        let sources = Map.empty
        // Map techniques to shader files
        sources
        |> Map.add (Sprite, Set.empty) { VertexPath = "Assets/Shaders/basic.vert"; FragmentPath = "Assets/Shaders/basic.frag" }
        //|> Map.add (Sprite, Set.singleton "Instanced") { VertexPath = "Assets/Shaders/sprite_instanced.vert"; FragmentPath = "Assets/Shaders/sprite.frag" }
        //|> Map.add (FullscreenPost, Set.empty) { VertexPath = "Assets/Shaders/fullscreen.vert"; FragmentPath = "Assets/Shaders/post.frag" }
        //|> Map.add (Custom "BlackSquare", Set.empty) { VertexPath = "Assets/Shaders/basic.vert"; FragmentPath = "Assets/Shaders/basic.frag" }
    
    // Creates a fresh executor state with an empty cache and no geometry.
    // We need a clean slate for each rendering session to avoid carrying over old data, setting up the cache and buffers so we’re ready to process the graph.
      // I guess the proper technique is to make things pink so they stand out when shader/material fails? This is for the future.
    let createState () = {
        Cache = { 
            Shaders = Map.empty
            Materials = Map.empty
            ShaderSources = initializeShaderSources ()
        }
        FrameResources = Map.empty
        QuadVao = None
        QuadVbo = None
        QuadEbo = None
    }
    
    /// Sets up a quad’s geometry (vertices and indices) for 2D rendering, stored in OpenGL buffers.
    // Quads are the backbone of 2D rendering for sprites and UI, so we initialize them once and reuse them to save GPU time.
    // It creates vertex and index buffers (VAO, VBO, EBO) and sets up attributes for positions and texture coordinates.
    let initializeQuadGeometry (state: ExecutorState byref) =
        match state.QuadVao with
        | Some _ -> () // Already initialized
        | None ->
            // Quad vertices (position + texcoord)
            let vertices = [|
                // Position + TexCoord
                -0.5f; -0.5f; 0.0f; 1.0f; // Bottom-left
                 0.5f; -0.5f; 1.0f; 1.0f; // Bottom-right
                 0.5f;  0.5f; 1.0f; 0.0f; // Top-right
                -0.5f;  0.5f; 0.0f; 0.0f  // Top-left
            |]
            
            let indices = [| 0u; 1u; 2u; 2u; 3u; 0u |]
            
            // Generate VAO
            let vaos = Array.zeroCreate<uint32> 1
            OpenGL.Gl.GenVertexArrays(vaos)
            let vao = vaos.[0]
            OpenGL.Gl.BindVertexArray(vao)
            
            // Generate VBO
            let vbos = Array.zeroCreate<uint32> 1
            OpenGL.Gl.GenBuffers(vbos)
            let vbo = vbos.[0]
            OpenGL.Gl.BindBuffer(OpenGL.BufferTarget.ArrayBuffer, vbo)
            OpenGL.Gl.BufferData(OpenGL.BufferTarget.ArrayBuffer, uint32 (vertices.Length * sizeof<single>), vertices, OpenGL.BufferUsage.StaticDraw)
            
            // Generate EBO
            let ebos = Array.zeroCreate<uint32> 1
            OpenGL.Gl.GenBuffers(ebos)
            let ebo = ebos.[0]
            OpenGL.Gl.BindBuffer(OpenGL.BufferTarget.ElementArrayBuffer, ebo)
            OpenGL.Gl.BufferData(OpenGL.BufferTarget.ElementArrayBuffer, uint32 (indices.Length * sizeof<uint>), indices, OpenGL.BufferUsage.StaticDraw)
            
            // Setup vertex attributes
            OpenGL.Gl.VertexAttribPointer(0u, 2, OpenGL.VertexAttribPointerType.Float, false, 4 * sizeof<single>, IntPtr.Zero)
            OpenGL.Gl.EnableVertexAttribArray(0u)
            OpenGL.Gl.VertexAttribPointer(1u, 2, OpenGL.VertexAttribPointerType.Float, false, 4 * sizeof<single>, IntPtr(2 * sizeof<single>))
            OpenGL.Gl.EnableVertexAttribArray(1u)
            
            // Unbind
            OpenGL.Gl.BindBuffer(OpenGL.BufferTarget.ArrayBuffer, 0u)
            OpenGL.Gl.BindVertexArray(0u)
            
            // Store in state
            state <- { state with QuadVao = Some vao; QuadVbo = Some vbo; QuadEbo = Some ebo }
            OpenGL.Hl.Assert()
    
    // Loads vertex and fragment shader source from files or falls back to defaults.
    // This makes sure we always have shader code to work with, even if files are missing, which keeps rendering robust during development or on different systems.
    // It tries to read GLSL files from disk and uses simple default shaders if they’re not found.
    let private loadShaderFromFiles vertexPath fragmentPath =
        try
            // Try to load .glsl files
            let vertexSource = 
                if File.Exists(vertexPath) then
                    File.ReadAllText(vertexPath)
                else
                    // Fallback to default vertex shader
                    """#version 330
                    layout(location = 0) in vec2 aPosition;
                    layout(location = 1) in vec2 aTexCoord;
                    out vec2 vTexCoord;
                    uniform mat4 uMVP;
                    void main() {
                        vTexCoord = aTexCoord;
                        gl_Position = uMVP * vec4(aPosition, 0.0, 1.0);
                    }"""
            
            let fragmentSource = 
                if File.Exists(fragmentPath) then
                    File.ReadAllText(fragmentPath)
                else
                    // Fallback to default fragment shader
                    """#version 330
                    in vec2 vTexCoord;
                    out vec4 FragColor;
                    uniform vec4 uColor;
                    void main() {
                        FragColor = uColor;
                    }"""
            
            Some (vertexSource, fragmentSource)
        with ex ->
            Log.error $"Failed to load shader files {vertexPath}, {fragmentPath}: {ex.Message}"
            None
    
    // Gets or creates a shader for a given technique and features, caching it for reuse.
    // Compiling shaders is a heavy operation, so we cache them by a unique key (technique + features) to avoid doing it every frame, which keeps our game running smoothly.
    // It looks up the right shader files, compiles them, and stores the result in the cache.
    let private getOrCreateShaderFromTechnique technique features (state: ExecutorState byref) =
        let cacheKey = $"""{technique}_{features |> Set.toList |> String.concat "_"}"""
        
        match Map.tryFind cacheKey state.Cache.Shaders with
        | Some shader -> shader
        | None ->
            // Find shader source paths for this technique
            let shaderSourceOpt = 
                state.Cache.ShaderSources 
                |> Map.tryFind (technique, features)
                |> Option.orElse (
                    // Try without features as fallback
                    state.Cache.ShaderSources |> Map.tryFind (technique, Set.empty)
                )
            
            match shaderSourceOpt with
            | Some source ->
                match loadShaderFromFiles source.VertexPath source.FragmentPath with
                | Some (vertexSource, fragmentSource) ->
                    // Compile shader
                    let programId = OpenGL.Shader.CreateShaderFromStrs(vertexSource, fragmentSource)
                    let uniformLocations = Map.empty // TODO: Query uniform locations
                    let runtimeShader = RuntimeShader (programId, uniformLocations)
                    
                    // Cache it
                    let updatedShaders = Map.add cacheKey runtimeShader state.Cache.Shaders
                    let updatedCache = { state.Cache with Shaders = updatedShaders }
                    state <- { state with Cache = updatedCache }
                    
                    runtimeShader
                | None ->
                    RuntimeShader (0u, Map.empty)
            | None ->
                Log.warn $"No shader source found for technique {technique} with features {features}"
                RuntimeShader (0u, Map.empty)
    
    // Processes a resource node, turning graph resources into actual OpenGL objects.
    // This sets up the GPU resources (like shaders or textures) that actions will use, ensuring everything’s ready before we start drawing.
    // It handles each resource type differently—compiling shaders, pairing materials with shaders, or pulling textures from Nu’s asset system.
    let executeResourceNode resourceId resourceNode (assetResolver: AssetResolver) (state: ExecutorState byref) =
        match resourceNode with
        | TextureResource (textureType, _) ->
            // Output textures don't need processing - we render to Nu's framebuffer
            ()
            
        | ShaderResource (shaderType, _) ->
            let shader = getOrCreateShaderFromTechnique shaderType.Technique shaderType.Features &state
            state <- { state with FrameResources = Map.add resourceId shader state.FrameResources }
        
        | MaterialResource (materialType, _) ->
            let shaderId = materialType.MaterialInstance.ShaderId
            match Map.tryFind shaderId state.FrameResources with
            | Some (RuntimeShader (programId, uniformLocations)) ->
                let material = RuntimeMaterial (programId, materialType.MaterialInstance.Uniforms, materialType.MaterialInstance.Textures)
                state <- { state with FrameResources = Map.add resourceId material state.FrameResources }
            | _ -> 
                Log.warn $"Material {resourceId} references non-existent shader {shaderId}"
        
        | ImportedTextureResource imageTag ->
            // Use Nu's asset resolver
            match assetResolver imageTag with
            | ValueSome (TextureAsset texture) -> 
                let runtimeTexture = RuntimeTexture (texture.TextureId, int texture.TextureMetadata.TextureWidth, int texture.TextureMetadata.TextureHeight)
                state <- { state with FrameResources = Map.add resourceId runtimeTexture state.FrameResources }
            | _ -> 
                Log.warn $"Failed to resolve imported texture {imageTag}"
        
        | _ -> 
            Log.info $"Resource type not implemented yet: {resourceNode}"
    
    // Applies blending settings for a draw call, like transparency or additive effects.
    // This controls how new pixels blend with what’s already on the screen, which is crucial for effects like see-through sprites or glowing particles.
    // It translates our graph’s blend settings into OpenGL’s blend functions and equations.
    let private setBlendState (blendStateOpt: BlendState option) =
        match blendStateOpt with
        | Some blendState ->
            OpenGL.Gl.Enable(OpenGL.EnableCap.Blend)
            let toGLBlendFactor = function
                | Zero -> OpenGL.BlendingFactor.Zero
                | One -> OpenGL.BlendingFactor.One
                | SrcColor -> OpenGL.BlendingFactor.SrcColor
                | OneMinusSrcColor -> OpenGL.BlendingFactor.OneMinusSrcColor
                | DstColor -> OpenGL.BlendingFactor.DstColor
                | OneMinusDstColor -> OpenGL.BlendingFactor.OneMinusDstColor
                | SrcAlpha -> OpenGL.BlendingFactor.SrcAlpha
                | OneMinusSrcAlpha -> OpenGL.BlendingFactor.OneMinusSrcAlpha
                | DstAlpha -> OpenGL.BlendingFactor.DstAlpha
                | OneMinusDstAlpha -> OpenGL.BlendingFactor.OneMinusDstAlpha
            
            let toGLBlendOp = function
                | Add -> OpenGL.BlendEquationMode.FuncAdd
                | Subtract -> OpenGL.BlendEquationMode.FuncSubtract
                | ReverseSubtract -> OpenGL.BlendEquationMode.FuncReverseSubtract
                | Min -> OpenGL.BlendEquationMode.Min
                | Max -> OpenGL.BlendEquationMode.Max
            
            OpenGL.Gl.BlendFuncSeparate(
                toGLBlendFactor blendState.ColorSrc,
                toGLBlendFactor blendState.ColorDst,
                toGLBlendFactor blendState.AlphaSrc,
                toGLBlendFactor blendState.AlphaDst)
            
            OpenGL.Gl.BlendEquationSeparate(
                toGLBlendOp blendState.ColorOp,
                toGLBlendOp blendState.AlphaOp)
        | None ->
            OpenGL.Gl.Disable(OpenGL.EnableCap.Blend)
    
    // Executes a shader action, handling the full draw process for a material and geometry.
    // This is the core of our rendering, taking a shader action (like drawing a sprite) and making it happen on the GPU with all the right settings—transforms, textures, and blending.
    // It sets up the shader, applies uniforms and textures, computes the model-view-projection matrix for 2D rendering, and draws the geometry.
    let executeShaderAction actionId shaderAction (assetResolver: AssetResolver) eyeCenter eyeSize viewport (state: ExecutorState byref) =
        match Map.tryFind shaderAction.Material.Handle.Id state.FrameResources with
        | Some (RuntimeMaterial (programId, uniforms, textures)) ->
            if programId <> 0u then
                // Use shader program
                OpenGL.Gl.UseProgram(programId)
                
                // Find transform in uniforms and build model matrix
                let mutable modelMatrix = Matrix4x4.Identity
                let mutable transformAbsolute = false
                
                // Process transform first
                for KeyValue(name, uniformValue) in uniforms do
                    match uniformValue with
                    | StaticTransform transform ->
                        transformAbsolute <- transform.Absolute
                        let scaledPosition = transform.Position * single viewport.DisplayScalar
                        let scaledSize = transform.Size * single viewport.DisplayScalar
                        modelMatrix <- Matrix4x4.CreateAffine(scaledPosition, transform.Rotation, scaledSize)
                    | _ -> ()
                
                // Set regular uniforms
                for KeyValue(name, uniformValue) in uniforms do
                    let location = OpenGL.Gl.GetUniformLocation(programId, name)
                    if location <> -1 then
                        match uniformValue with
                        | StaticTransform _ -> () // Already processed
                        | StaticFloat f -> OpenGL.Gl.Uniform1(location, f)
                        | StaticVec2 v -> OpenGL.Gl.Uniform2(location, v.X, v.Y)
                        | StaticVec3 v -> OpenGL.Gl.Uniform3(location, v.X, v.Y, v.Z)
                        | StaticVec4 v -> OpenGL.Gl.Uniform4(location, v.X, v.Y, v.Z, v.W)
                        | StaticColor c -> OpenGL.Gl.Uniform4(location, c.R, c.G, c.B, c.A)
                        | StaticMatrix m -> 
                            let matrixArray = m.ToArray()
                            OpenGL.Gl.UniformMatrix4(location, false, matrixArray)
                        | _ -> 
                            Log.warn $"Dynamic uniform '{name}' type not supported yet"
                
                // Calculate and set MVP matrix
                let viewProjection = Viewport.getViewProjection2d transformAbsolute eyeCenter eyeSize viewport
                let mvp = modelMatrix * viewProjection
                let mvpLocation = OpenGL.Gl.GetUniformLocation(programId, "uMVP")
                if mvpLocation <> -1 then
                    let mvpArray = mvp.ToArray()
                    OpenGL.Gl.UniformMatrix4(mvpLocation, false, mvpArray)
                
                // Bind textures
                let mutable textureUnit = 0
                for KeyValue(name, imageTag) in textures do
                    match assetResolver imageTag with
                    | ValueSome (TextureAsset texture) ->
                        let location = OpenGL.Gl.GetUniformLocation(programId, name)
                        if location <> -1 then
                            OpenGL.Gl.ActiveTexture(OpenGL.TextureUnit.Texture0 + enum<OpenGL.TextureUnit> textureUnit)
                            OpenGL.Gl.BindTexture(OpenGL.TextureTarget.Texture2d, texture.TextureId)
                            OpenGL.Gl.Uniform1(location, textureUnit)
                            textureUnit <- textureUnit + 1
                    | _ -> ()
                
                // Set render state
                let materialResource = shaderAction.Material.Handle.ResourceType
                setBlendState materialResource.MaterialInstance.RenderState.Blending
                
                // Draw geometry
                match shaderAction.Geometry with
                | Quad ->
                    match state.QuadVao with
                    | Some vao ->
                        OpenGL.Gl.BindVertexArray(vao)
                        OpenGL.Gl.DrawElements(OpenGL.PrimitiveType.Triangles, 6, OpenGL.DrawElementsType.UnsignedInt, IntPtr.Zero)
                        OpenGL.Gl.BindVertexArray(0u)
                    | None -> Log.warn "Quad geometry not initialized"
                | FullscreenTriangle ->
                    // TODO: Implement fullscreen triangle
                    ()
                | _ ->
                    Log.warn $"Geometry type {shaderAction.Geometry} not implemented"
                
                // Unbind shader
                OpenGL.Gl.UseProgram(0u)
                OpenGL.Hl.Assert()
        | _ -> 
            Log.warn $"Shader action {actionId} references invalid material"
    
    // Executes an action node, deciding what to do based on the action type.
    // This is the central hub that routes each action to the right handler, whether it’s drawing with a shader, clearing a buffer, or presenting the final image.
    // It’s critical for keeping the rendering pipeline organized, ensuring each action type is processed correctly without mixing them up.
    let executeActionNode actionId actionNode (assetResolver: AssetResolver) eyeCenter eyeSize viewport (state: ExecutorState byref) =
        match actionNode with
        | ShaderAction shaderAction -> executeShaderAction actionId shaderAction assetResolver eyeCenter eyeSize viewport &state
        | ClearAction clearAction -> 
            OpenGL.Gl.ClearColor(clearAction.Color.R, clearAction.Color.G, clearAction.Color.B, clearAction.Color.A)
            OpenGL.Gl.Clear(OpenGL.ClearBufferMask.ColorBufferBit)
        | PresentAction presentAction -> 
            () // Present is handled by Nu's renderer
        | ComputeAction computeAction -> 
            Log.info $"Compute action {actionId} not implemented yet"
    
    // Main function to execute the entire render graph, turning nodes into OpenGL calls.
    // This is where everything comes together—taking the whole graph and making it show up on screen by processing resources, actions, and sub-graphs in the right order.
    // It uses topological sorting to respect dependencies, ensuring we don’t try to draw before resources are ready.
    let rec execute graph eyeCenter eyeSize viewport (assetResolver: AssetResolver) (state: ExecutorState byref) =
        // Initialize geometry if needed
        initializeQuadGeometry &state
        
        // Clear frame resources
        state <- { state with FrameResources = Map.empty }
        
        // Get execution order
        let executionOrder = RenderGraph.topologicalSort graph
        
        // Execute resources first, then actions
        for nodeId in executionOrder do
            match Map.tryFind nodeId graph.Nodes with
            | Some node ->
                match node with
                | Resource (resourceId, resourceNode) ->
                    executeResourceNode resourceId resourceNode assetResolver &state
                | Action (actionId, actionNode) ->
                    executeActionNode actionId actionNode assetResolver eyeCenter eyeSize viewport &state
                | Pass (passId, passDesc) ->
                    // Stub for render passes
                    Log.info $"Render pass '{passDesc.Name}' execution not implemented yet"
                | SubGraph (subGraphId, subGraph) ->
                    // Recursive execution
                    execute subGraph eyeCenter eyeSize viewport assetResolver &state
            | None -> ()
        
        OpenGL.Hl.Assert()
    
    // Cleans up the executor state, freeing GPU resources like shaders and geometry.
    // This prevents memory leaks by deleting all OpenGL objects when we’re done, ensuring we don’t leave the GPU in a messy state.
    // It clears the shader cache, deletes quad buffers, and resets the state to a clean slate.
    let cleanup (state: ExecutorState byref) =
        // Clean up cached shaders
        for KeyValue(_, resource) in state.Cache.Shaders do
            match resource with
            | RuntimeShader (programId, _) when programId <> 0u -> OpenGL.Gl.DeleteProgram(programId)
            | _ -> ()
        
        // Clean up quad geometry
        match state.QuadVao with
        | Some vao -> OpenGL.Gl.DeleteVertexArrays([|vao|])
        | None -> () 
        
        match state.QuadVbo with
        | Some vbo -> OpenGL.Gl.DeleteBuffers([|vbo|])
        | None -> ()
        
        match state.QuadEbo with
        | Some ebo -> OpenGL.Gl.DeleteBuffers([|ebo|])
        | None -> ()
        
        state <- createState()
        
        OpenGL.Hl.Assert()