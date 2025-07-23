namespace MyGame
open System
open System.Numerics
open Prime
open Nu
open RenderGraph




type BlackSquareFacet() =
    inherit Facet(false, false, false)
    
    override this.Render(renderPass, entity, world) =
        if renderPass = NormalPass then
            let transform = entity.GetTransform world
            
            let uniforms = Map.ofList [
                "uTransform", StaticTransform transform
                "uColor", StaticColor Color.Black
            ]
            
            let textures = Map.empty
            
            let graph = 
                RenderGraphBuilder.createSpriteGraph 
                    "BlackSquare" 
                    "Assets/Shaders/basic.vert"
                    "Assets/Shaders/basic.frag"
                    uniforms 
                    textures 
                    None 
                    None
            
            let elevation = entity.GetElevationLocal world
            let dummyAsset = Assets.Default.Image
            
            World.enqueueRenderMessage2d (ExecuteRenderGraph2d (graph, elevation, 0.0f, dummyAsset)) world
    
    override this.GetAttributesInferred(entity, world) =
        AttributesInferred.important (entity.GetSize world) v3Zero


type BlackSquareDispatcher() =
    inherit EntityDispatcher(true, false, false, false)
    
    static member Facets = [typeof<BlackSquareFacet>]
    static member Properties = []

module RenderGraphHelpers =
    let doBlackSquare name args world =
        World.doEntity<BlackSquareDispatcher> name args world