module MyGame.RenderGraphDynamicSpriteFacet

open System
open System.Numerics
open Prime
open Nu
open RenderGraph

// Store a function that produces uniforms given the world
type Entity with
    member this.GetUniformProducer world : (World -> Map<string, UniformValue>) = 
        this.Get (nameof this.UniformProducer) world
    member this.SetUniformProducer value world = 
        this.Set (nameof this.UniformProducer) value world
    member this.UniformProducer = 
        lens (nameof this.UniformProducer) this this.GetUniformProducer this.SetUniformProducer

type DynamicShaderDispatcher() =
    inherit EntityDispatcher(true, false, false, false)
    
    static member Facets = [typeof<DynamicShaderFacet>]
    static member Properties = 
        [define Entity.Color Color.One
         define Entity.UniformProducer (fun _ -> Map.empty)]

and DynamicShaderFacet() =
    inherit Facet(false, false, false)
    
    override this.Render(renderPass, entity, world) =
        if renderPass = NormalPass then
            let transform = entity.GetTransform world
            
            // Get the uniform producer function and evaluate it with current world
            let uniformProducer = entity.GetUniformProducer world
            let dynamicUniforms = uniformProducer world
            
            // Standard uniforms
            let standardUniforms = Map.ofList [
                "uTransform", StaticTransform transform
            ]
            
            // Merge uniforms (dynamic overrides standard)
            let uniforms = Map.fold (fun acc k v -> Map.add k v acc) standardUniforms dynamicUniforms
            
            let textures = Map.empty
            
            // Use basicv2 shaders that we know work
            let graph = 
                RenderGraphBuilder.createSpriteGraphWithSource 
                    "DynamicShader" 
                    "Assets/Shaders/basicv2.vert"
                    "Assets/Shaders/basicv2.frag"
                    uniforms 
                    textures 
                    None 
                    None
            
            let elevation = entity.GetElevationLocal world
            let dummyAsset = Assets.Default.Image
            
            World.enqueueRenderMessage2d (ExecuteRenderGraph2d (graph, elevation, 0.0f, dummyAsset)) world
    
    override this.GetAttributesInferred(entity, world) =
        AttributesInferred.important (entity.GetSize world) v3Zero

module RenderGraphDynamicHelpers =
    
    /// Create an entity with dynamic uniforms
    let doDynamicShader name uniformProducer entityArgs world =
        // Add the uniform producer to the entity arguments
        let enhancedArgs = 
            (Entity.UniformProducer .= uniformProducer) :: entityArgs
        
        // Create the entity with the dynamic dispatcher
        World.doEntity<DynamicShaderDispatcher> name enhancedArgs world
    
    /// Create a uniform producer for rainbow animation
    let createRainbowProducer (timeFunc: World -> single) speed color : (World -> Map<string, UniformValue>) =
        fun world ->
            Map.ofList [
                "uColor", StaticColor color
                "uTime", StaticFloat (timeFunc world)  // Evaluate the function!
                "uSpeed", StaticFloat speed
            ]