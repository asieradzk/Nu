//all this is a temporary bypass for traditional rendering
module MyGame.RenderGraphSpriteFacet

open System
open System.Numerics
open Prime
open Nu
open RenderGraph
open MyGame.ShaderTypes
open MyGame

//entity extension with "ShaderPackage" - I don't like the name 100% suggestions welcome
type Entity with

    // Uses concrete ShaderPackage type to maintain type information
    member this.GetShaderPackage world : ShaderPackage = 
        match this.TryGetProperty "ShaderPackage" world with
        | Some property -> property.PropertyValue :?> ShaderPackage
        | None -> TestBasicShader.black  // Default fallback
    member this.SetShaderPackage value world = this.Set "ShaderPackage" value world
    member this.ShaderPackage = lens "ShaderPackage" this this.GetShaderPackage this.SetShaderPackage

// Generic shader facet that uses Entity.ShaderPackage property
type ShaderFacet() =
    inherit Facet(false, false, false)
    
    override this.Render(renderPass, entity, world) =
        if renderPass = NormalPass then 
            try
                // Check if shader package property exists
                match entity.TryGetProperty "ShaderPackage" world with
                | Some property ->
                    // Get shader package from entity property
                    let shaderPackage = property.PropertyValue :?> ShaderPackage
                    let transform = entity.GetTransform world
                    
                    //the pre-built extract data function to avoid runtime reflection
                    //we want to know the source of shader program and the uniform magic strings
                    let (source, uniformsArray) = shaderPackage.ExtractData()
                    
                    //shader source per platform, pre-determined constraints
                    let (vertPath, fragPath) = 
                        match source with
                        | ShaderSource.SeparateFiles(v, f) -> (v, f)
                        | ShaderSource.SingleGlslFile(path) -> (path, path)
                        | ShaderSource.InlineSource(_, _) -> 
                            ("Assets/Shaders/basicv2.vert", "Assets/Shaders/basicv2.frag") // fallback
                    
                    //mapping uniforms to magic strings
                    let uniforms = 
                        let uniformsList = uniformsArray |> Array.toList
                        let withTransform = ("uTransform", StaticTransform transform) :: uniformsList
                        withTransform |> Map.ofList
                    
                    //create partial graph and submit to render graph
                    let graph = 
                        RenderGraphBuilder.createSpriteGraphWithSource 
                            "ShaderEntity" 
                            vertPath
                            fragPath
                            uniforms 
                            Map.empty  // textures - this is TODO
                            None       // TODO
                            None       // TODOs
                    
                    let elevation = entity.GetElevationLocal world
                    let dummyAsset = Assets.Default.Image
                    World.enqueueRenderMessage2d (ExecuteRenderGraph2d (graph, elevation, 0.0f, dummyAsset)) world
                    //in the final version we just send partial graph directly.
                | None ->
                    // No shader package property - entity won't render
                    ()
            with
            | ex -> 
                // Log error but don't crash - entity just won't render
                // We Probably want a fallback shader to render stuff pink
                
                Log.info $"ShaderFacet render error: {ex.Message}"
                Log.info $"Stack trace: {ex.StackTrace}"
    
    override this.GetAttributesInferred(entity, world) =
        AttributesInferred.important (entity.GetSize world) v3Zero

//Dispatcher that uses ShaderFacet for module-based shader rendering
type ShaderDispatcher() =
    inherit EntityDispatcher(true, false, false, false)
    
    static member Facets = [typeof<ShaderFacet>]
    
    static member Properties =
        [define Entity.ShaderPackage TestBasicShader.black]


// just a helper in final version we want to just use World.doEntity, doPanel etc
module RenderGraphHelpers =
    let doEntityShader name entityArgs world =
        World.doEntity<ShaderDispatcher> name entityArgs world
    
